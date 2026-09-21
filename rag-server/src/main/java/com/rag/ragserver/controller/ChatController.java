package com.rag.ragserver.controller;


import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.rag.ragserver.common.R;
import com.rag.ragserver.domain.*;
import com.rag.ragserver.dto.ChatStart;
import com.rag.ragserver.dto.ChatStream;
import com.rag.ragserver.dto.MessageEditDTO;
import com.rag.ragserver.dto.MessageRetryDTO;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.service.*;
import com.rag.ragserver.utils.ChatStreamRegistry;
import com.rag.ragserver.utils.ModelUtils;
import com.rag.ragserver.domain.model.vo.ModelPermission;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import javax.servlet.http.HttpServletRequest;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.Base64;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
@RestController
@RequestMapping("/chat")
@RequiredArgsConstructor
public class ChatController {
    private final HttpServletRequest request;
    private final ModelUtils modelUtils;
    private final QuerySessionsService querySessionsService;
    private final ConversationMessagesService conversationMessagesService;
    private final KnowledgeBasesService knowledgeBasesService;
    private final KbPermissionService kbPermissionService;
    private final RolesService rolesService;
    private final RequestLimitationsService requestLimitationsService;
    private final WebClient webClient;
    private final ChatStreamRegistry chatStreamRegistry;

    @PostMapping("/start")
    public R<Map<String, Long>> startChat(@RequestBody ChatStart chatStart) {
        Long userId = (Long) request.getAttribute("userId");
        Integer roleId = (Integer) request.getAttribute("roleId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");

        validatePermissions(roleId, chatStart.getModelId(), chatStart.getKbId(), userId, workspaceId);
        Long kbId = chatStart.getKbId();

        QuerySessions querySession = new QuerySessions();
        querySession.setUserId(userId);
        querySession.setWorkspaceId(workspaceId);
        querySessionsService.save(querySession);
        // 保存对话
        ConversationMessages conversationMessage = new ConversationMessages();
        conversationMessage.setKbId(kbId);
        conversationMessage.setSessionId(querySession.getId());
        conversationMessage.setUserId(userId);
        conversationMessage.setRole("user");
        conversationMessage.setContent(chatStart.getQuestion());
        conversationMessage.setModelId(chatStart.getModelId());
        conversationMessage.setStatus("pending");

        // 保存 options
        if (chatStart.getOptions() != null) {
            Map<String, Object> opts = chatStart.getOptions();
            if (opts.containsKey("thinking") && Boolean.FALSE.equals(opts.get("thinking"))) {
                opts.remove("thinking");
            }
            conversationMessage.setOptions(opts);
        }

        conversationMessagesService.save(conversationMessage);

        return R.success(Map.of("sessionId", querySession.getId()));
    }

    @GetMapping("/sessions/{sessionId}/messages")
    public R getMessages(@PathVariable Long sessionId) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        // 1. 判断该session是否存在于该用户的某个工作空间下
        LambdaQueryWrapper<QuerySessions> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(QuerySessions::getId, sessionId)
                .eq(QuerySessions::getUserId, userId)
                .eq(QuerySessions::getWorkspaceId, workspaceId);
        QuerySessions querySession = querySessionsService.getOne(queryWrapper);
        if (querySession == null) {
            throw new BusinessException(404, "会话不存在");
        }
        // 2. 获取该session下的所有未删除消息
        LambdaQueryWrapper<ConversationMessages> messageQueryWrapper = new LambdaQueryWrapper<>();
        messageQueryWrapper
                .select(
                        ConversationMessages::getId,
                        ConversationMessages::getSessionId,
                        ConversationMessages::getUserId,
                        ConversationMessages::getRole,
                        ConversationMessages::getContent,
                        ConversationMessages::getModelId,
                        ConversationMessages::getKbId,
                        ConversationMessages::getStatus,
                        ConversationMessages::getCreatedAt,
                        ConversationMessages::getLatencyMs,
                        ConversationMessages::getPromptTokens,
                        ConversationMessages::getCompletionTokens,
                        ConversationMessages::getTotalTokens,
                        ConversationMessages::getOptions,
                        ConversationMessages::getThinking
                )
                .eq(ConversationMessages::getSessionId, sessionId)
                .and(w -> w.isNull(ConversationMessages::getIsDeleted).or().eq(ConversationMessages::getIsDeleted, 0))
                .orderByAsc(ConversationMessages::getCreatedAt)
                .orderByAsc(ConversationMessages::getId);
        List<ConversationMessages> messages = conversationMessagesService.list(messageQueryWrapper);

        // 3. 单独查询最后5条assistant消息的 rag_context，避免加载历史消息的大字段
        List<ConversationMessages> lastNRagMessages = conversationMessagesService.getLastNRagContextMessages(sessionId, 5);
        // 注意last是倒序的
        if (!lastNRagMessages.isEmpty()) {
            int a = messages.size() - 1;
            int b = 0;

            while (b < lastNRagMessages.size()) {
                long currentMessageId = lastNRagMessages.get(b).getId();
                while (a >= 0 && !messages.get(a).getId().equals(currentMessageId)) {
                    a--;
                }
                if (a >= 0) {
                    messages.get(a).setRagContext(lastNRagMessages.get(b).getRagContext());
                } else {
                    break;
                }
                b++;
            }
        }
        return R.success(messages);
    }

    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> streamChat(@RequestBody ChatStream chatStream) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        Integer roleId = (Integer) request.getAttribute("roleId");

        ModelPermission modelPermission = validatePermissions(roleId, chatStream.getModelId(), chatStream.getKbId(), userId, workspaceId);

        Long sessionId = chatStream.getSessionId();
        List<ConversationMessages> messageList = getSessionMessages(sessionId, userId);

        ConversationMessages lastMessage = messageList.get(messageList.size() - 1);
        Long currentUserMessageId = processNewMessage(chatStream, messageList, lastMessage, userId);

        return executeStreamChat(chatStream, userId, modelPermission, messageList, currentUserMessageId);
    }

    private void markMessageGenerating(Long sessionId, Long messageId) {
        Mono.fromRunnable(() -> {
            // 状态单调：只允许 pending -> generating，避免终态被并发回退
            boolean updated = conversationMessagesService.update(
                    new LambdaUpdateWrapper<ConversationMessages>()
                            .eq(ConversationMessages::getId, messageId)
                            .eq(ConversationMessages::getStatus, "pending")
                            .set(ConversationMessages::getStatus, "generating")
            );
            if (!updated) {
                log.warn("消息状态未推进为 generating（可能已是终态）, messageId={}", messageId);
            }

            // 同时更新 Session 的最后活跃时间
            querySessionsService.update(
                    new LambdaUpdateWrapper<QuerySessions>()
                            .eq(QuerySessions::getId, sessionId)
                            .set(QuerySessions::getLastActiveAt, new Date())
            );
        }).subscribeOn(Schedulers.boundedElastic()).subscribe();
    }

    @PostMapping(value = "/messages/{messageId}/edit", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> editAndRegenerate(@PathVariable Long messageId, @RequestBody MessageEditDTO dto) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        Integer roleId = (Integer) request.getAttribute("roleId");

        ModelPermission modelPermission = validatePermissions(roleId, dto.getModelId(), dto.getKbId(), userId, workspaceId);

        // 更新用户消息内容及options
        ConversationMessages userMsg = conversationMessagesService.getById(messageId);
        if (userMsg != null && userMsg.getUserId().equals(userId)) {
            userMsg.setContent(dto.getNewContent());
            if (dto.getOptions() != null) {
                Map<String, Object> opts = dto.getOptions();
                if (opts.containsKey("thinking") && Boolean.FALSE.equals(opts.get("thinking"))) {
                    opts.remove("thinking");
                }
                userMsg.setOptions(opts);
            }
            conversationMessagesService.updateById(userMsg);
        }
        conversationMessagesService.editLastUserMessage(dto.getSessionId(), messageId, userId, dto.getNewContent());

        ChatStream chatStream = new ChatStream();
        chatStream.setSessionId(dto.getSessionId());
        chatStream.setModelId(dto.getModelId());
        chatStream.setKbId(dto.getKbId());
        chatStream.setQuestion(dto.getNewContent());
        chatStream.setOptions(dto.getOptions());

        List<ConversationMessages> messageList = getSessionMessages(dto.getSessionId(), userId);
        Long currentUserMessageId = messageList.get(messageList.size() - 1).getId();

        return executeStreamChat(chatStream, userId, modelPermission, messageList, currentUserMessageId);
    }

    @PostMapping(value = "/messages/{userMessageId}/retry", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> retryRegenerate(@PathVariable Long userMessageId, @RequestBody MessageRetryDTO dto) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        Integer roleId = (Integer) request.getAttribute("roleId");

        ModelPermission modelPermission = validatePermissions(roleId, dto.getModelId(), dto.getKbId(), userId, workspaceId);

        // 如果提供了新的 options，更新原用户消息的 options
        if (dto.getOptions() != null) {
            ConversationMessages userMsg = conversationMessagesService.getById(userMessageId);
            if (userMsg != null) {
                Map<String, Object> opts = dto.getOptions();
                if (opts.containsKey("thinking") && Boolean.FALSE.equals(opts.get("thinking"))) {
                    opts.remove("thinking");
                }
                userMsg.setOptions(opts);
                conversationMessagesService.updateById(userMsg);
            }
        }

        conversationMessagesService.retryLastAssistantMessage(dto.getSessionId(), userMessageId, userId);

        ChatStream chatStream = new ChatStream();
        chatStream.setSessionId(dto.getSessionId());
        chatStream.setModelId(dto.getModelId());
        chatStream.setKbId(dto.getKbId());
        chatStream.setQuestion(null);
        chatStream.setOptions(dto.getOptions());

        List<ConversationMessages> messageList = getSessionMessages(dto.getSessionId(), userId);
        Long currentUserMessageId = messageList.get(messageList.size() - 1).getId();

        return executeStreamChat(chatStream, userId, modelPermission, messageList, currentUserMessageId);
    }

    private ModelPermission validatePermissions(Integer roleId, Long modelId, Long kbId, Long userId, Long workspaceId) {
        // Check daily token limit
        Roles role = rolesService.getById(roleId);
        if (role != null && role.getDailyMaxTokens() != null && role.getDailyMaxTokens() > 0) {
            Long todayUsage = conversationMessagesService.countTodayTokens(userId);
            if (todayUsage >= role.getDailyMaxTokens()) {
                throw new BusinessException(
                        403,
                        String.format("今日Token使用已达上限 (%d)，请明天再试或联系管理员升级", role.getDailyMaxTokens())
                );
            }
        }

        ModelPermission modelPermission = modelUtils.canUseModel(roleId, modelId);
        if (modelPermission == null) {
            throw new BusinessException(404, "您无权使用该模型");
        }
        if (kbId != null && !kbPermissionService.canReadKb(kbId, userId, workspaceId)) {
            throw new BusinessException(403, "没有权限访问该知识库");
        }
        return modelPermission;
    }

    private List<ConversationMessages> getSessionMessages(Long sessionId, Long userId) {
        LambdaQueryWrapper<ConversationMessages> messageQueryWrapper = new LambdaQueryWrapper<>();
        messageQueryWrapper
                .select(
                        ConversationMessages::getId,
                        ConversationMessages::getSessionId,
                        ConversationMessages::getRole,
                        ConversationMessages::getContent,
                        ConversationMessages::getRagContext,
                        ConversationMessages::getStatus,
                        ConversationMessages::getModelId,
                        ConversationMessages::getOptions,
                        ConversationMessages::getThinking,
                        ConversationMessages::getProviderResponseId,
                        ConversationMessages::getProviderResponseItems
                )
                .eq(ConversationMessages::getSessionId, sessionId)
                .eq(ConversationMessages::getUserId, userId)
                .and(w -> w.isNull(ConversationMessages::getIsDeleted).or().eq(ConversationMessages::getIsDeleted, 0))
                .orderByAsc(ConversationMessages::getCreatedAt)
                .orderByAsc(ConversationMessages::getId);
        List<ConversationMessages> messageList = conversationMessagesService.list(messageQueryWrapper);
        if (messageList.isEmpty()) {
            throw new BusinessException(400, "会话不存在或无权限访问");
        }
        return messageList;
    }

    private Long processNewMessage(ChatStream chatStream, List<ConversationMessages> messageList,
                                   ConversationMessages lastMessage, Long userId) {
        String lastRole = (String) lastMessage.getRole();

        // 最后一条不是 user：上一轮已完整（或数据异常），开启新一轮
        if (!"user".equals(lastRole)) {
            return insertUserMessage(chatStream, messageList, userId);
        }

        // 最后一条是 user 且没有 assistant：上一轮未完成（崩溃/写入失败/中断）
        String lastStatus = lastMessage.getStatus() != null ? lastMessage.getStatus().toString() : "pending";
        if ("generating".equals(lastStatus)) {
            if (chatStreamRegistry.isActive(chatStream.getSessionId())) {
                throw new BusinessException(400, "AI正在生成回复，请稍后再试");
            }
            log.warn("检测到陈旧的 generating 状态，按中断处理并复用该消息, sessionId={}, messageId={}",
                    chatStream.getSessionId(), lastMessage.getId());
        }

        // 复用该 user 行：请求带问题则覆盖，否则沿用原内容（前端刷新后的续跑）
        String question = chatStream.getQuestion();
        if (question == null || question.isEmpty()) {
            question = lastMessage.getContent();
        }
        if (question == null || question.isEmpty()) {
            throw new BusinessException(400, "请求信息为空");
        }
        lastMessage.setContent(question);
        lastMessage.setStatus("pending");
        if (chatStream.getModelId() != null) {
            lastMessage.setModelId(chatStream.getModelId());
        }
        if (chatStream.getKbId() != null) {
            lastMessage.setKbId(chatStream.getKbId());
        }
        if (chatStream.getOptions() != null) {
            lastMessage.setOptions(normalizeOptions(chatStream.getOptions()));
        }
        conversationMessagesService.updateById(lastMessage);
        chatStream.setQuestion(question);
        return lastMessage.getId();
    }

    private Long insertUserMessage(ChatStream chatStream, List<ConversationMessages> messageList, Long userId) {
        if (chatStream.getQuestion() == null || chatStream.getQuestion().isEmpty()) {
            throw new BusinessException(400, "请求信息为空");
        }
        ConversationMessages newUserMessage = new ConversationMessages();
        newUserMessage.setSessionId(chatStream.getSessionId());
        newUserMessage.setUserId(userId);
        newUserMessage.setKbId(chatStream.getKbId());
        newUserMessage.setRole("user");
        newUserMessage.setContent(chatStream.getQuestion());
        newUserMessage.setModelId(chatStream.getModelId());
        newUserMessage.setStatus("pending");

        // 保存 options
        if (chatStream.getOptions() != null) {
            newUserMessage.setOptions(normalizeOptions(chatStream.getOptions()));
        }

        conversationMessagesService.save(newUserMessage);
        messageList.add(newUserMessage);
        return newUserMessage.getId();
    }

    private Map<String, Object> normalizeOptions(Map<String, Object> options) {
        Map<String, Object> opts = new java.util.HashMap<>(options);
        if (opts.containsKey("thinking") && Boolean.FALSE.equals(opts.get("thinking"))) {
            opts.remove("thinking");
        }
        return opts;
    }

    private Flux<String> executeStreamChat(ChatStream chatStream, Long userId, ModelPermission modelPermission,
                                           List<ConversationMessages> messageList, Long currentUserMessageId) {
        Long sessionId = chatStream.getSessionId();
        Long kbId = chatStream.getKbId();
        KnowledgeBases kb = knowledgeBasesService.getById(kbId);

        StringBuffer sb = new StringBuffer();
        StringBuffer thinkingSb = new StringBuffer(); // Add thinking buffer
        List<Map<String, Object>> ragProcessList = new java.util.ArrayList<>();
        Map<String, Object> usageInfo = new java.util.HashMap<>(); // Store usage info
        Map<String, Object> providerResponseInfo = new java.util.HashMap<>();
        ObjectMapper objectMapper = new ObjectMapper();
        AtomicBoolean saved = new AtomicBoolean(false);

        for (ConversationMessages message : messageList) {
            if ("assistant".equals(message.getRole()) && !chatStream.getModelId().equals(message.getModelId())) {
                message.setProviderResponseId(null);
                message.setProviderResponseItems(null);
            }
        }

        Map<String, Object> options = new java.util.HashMap<>();
        // 合并用户传递的 options
        if (chatStream.getOptions() != null) {
            Map<String, Object> userOpts = new java.util.HashMap<>(chatStream.getOptions());
            if (userOpts.containsKey("thinking") && Boolean.FALSE.equals(userOpts.get("thinking"))) {
                userOpts.remove("thinking");
            }
            options.putAll(userOpts);
        }

        if (kbId != null && kb != null) {
            options.put("userId", kb.getOwnerUserId());
            options.put("kbId", kbId);
            options.put("systemPrompt", kb.getSystemPrompt());
        }
        options.put("promptCacheKey", buildPromptCacheKey(userId, sessionId, chatStream.getModelId()));
        Map<String, Object> info = Map.of(
                "history", messageList,
                "model", modelPermission,
                "options", options
        );

        Flux<String> streamFlux = webClient.post()
                .uri("/rag/chat/stream")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(info)
                .accept(MediaType.TEXT_EVENT_STREAM)
                .retrieve()
                .bodyToFlux(new ParameterizedTypeReference<ServerSentEvent<String>>() {
                })
                .doOnSubscribe(a -> {
                    chatStreamRegistry.markActive(sessionId, currentUserMessageId);
                    markMessageGenerating(sessionId, currentUserMessageId);
                })
                .doOnNext(ignored -> chatStreamRegistry.refresh(sessionId))
                .concatMap(event -> processStreamEvent(event, sb, thinkingSb, ragProcessList, objectMapper, usageInfo, providerResponseInfo))
                .concatWith(saveCompletedMessage(sessionId, userId, chatStream, currentUserMessageId, sb, thinkingSb, ragProcessList, objectMapper, usageInfo, providerResponseInfo, saved))
                .doOnError(e -> {
                    if (saved.compareAndSet(false, true)) {
                        saveStoppedMessage(sessionId, userId, chatStream, currentUserMessageId, sb, thinkingSb, ragProcessList, objectMapper, usageInfo, false)
                                .subscribeOn(Schedulers.boundedElastic()).subscribe();
                    }
                })
                .doOnCancel(() -> {
                    if (saved.compareAndSet(false, true)) {
                        saveStoppedMessage(sessionId, userId, chatStream, currentUserMessageId, sb, thinkingSb, ragProcessList, objectMapper, usageInfo, true)
                                .subscribeOn(Schedulers.boundedElastic()).subscribe();
                    }
                })
                .doFinally(signal -> chatStreamRegistry.clear(sessionId));

        RequestLimitations requestLimitations = requestLimitationsService.getOne(
                new LambdaQueryWrapper<RequestLimitations>()
                        .eq(RequestLimitations::getUserId, userId)
                        .last("limit 1")
        );
        Integer delaySecond = requestLimitations == null ? null : requestLimitations.getDelaySecond();
        if (delaySecond != null && delaySecond > 0) {
            return streamFlux.delaySubscription(Duration.ofSeconds(delaySecond));
        }
        return streamFlux;
    }

    private String buildPromptCacheKey(Long userId, Long sessionId, Long modelId) {
        String source = userId + ":" + sessionId + ":" + modelId;
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(source.getBytes(StandardCharsets.UTF_8));
            return "chat:v1:" + Base64.getUrlEncoder().withoutPadding().encodeToString(digest);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }

    private Flux<String> processStreamEvent(ServerSentEvent<String> event, StringBuffer sb, StringBuffer thinkingSb,
                                             List<Map<String, Object>> ragProcessList, ObjectMapper objectMapper,
                                             Map<String, Object> usageInfo, Map<String, Object> providerResponseInfo) {
        String dataStr = event.data();
        if (dataStr == null || dataStr.isEmpty()) return Flux.empty();

        try {
            JsonNode data = objectMapper.readTree(dataStr);
            String type = data.path("type").asText();

            if ("content".equals(type) || "thinking".equals(type)) {
                try {
                    String payloadJson = data.path("payload").asText();
                    String text = objectMapper.readValue(payloadJson, String.class);
                    if ("content".equals(type)) {
                        sb.append(text);
                    } else {
                        thinkingSb.append(text);
                    }
                    return Flux.just(objectMapper.writeValueAsString(
                            Map.of("type", type, "content", text)
                    ));
                } catch (JsonProcessingException e) {
                    log.error("解析content payload失败: {}", e.getMessage());
                    return Flux.error(new BusinessException(400, "数据转换异常"));
                }
            } else if ("error".equals(type)) {
                try {
                    String payloadJson = data.path("payload").asText();
                    String text = objectMapper.readValue(payloadJson, String.class);
                    sb.append(text);
                    return Flux.just(objectMapper.writeValueAsString(
                            Map.of("type", "content", "content", text)
                    ));
                } catch (JsonProcessingException e) {
                    log.error("解析error payload失败: {}", e.getMessage());
                    return Flux.error(new BusinessException(400, "数据转换异常"));
                }
            } else if ("process".equals(type)) {
                String payloadJson = data.path("payload").asText();
                try {
                    Map processInfo = objectMapper.readValue(payloadJson, Map.class);
                    ragProcessList.add(processInfo);
                    return Flux.just(objectMapper.writeValueAsString(
                            Map.of("type", "process", "payload", processInfo)
                    ));
                } catch (JsonProcessingException e) {
                    log.error("转发检索过程信息失败: {}", e.getMessage());
                    return Flux.empty();
                }
            } else if ("rag_summary".equals(type)) {
                String payloadJson = data.path("payload").asText();
                try {
                    List summary = objectMapper.readValue(payloadJson, List.class);
                    ragProcessList.clear();
                    ragProcessList.addAll(summary);
                } catch (Exception e) {
                    log.error("处理RAG汇总信息失败", e);
                }
            } else if ("provider_response".equals(type)) {
                JsonNode payload = data.get("payload");
                if (payload != null && !payload.isNull()
                        && payload.has("responseId") && !payload.get("responseId").isNull()
                        && payload.has("status") && !payload.get("status").isNull()
                        && "completed".equals(payload.get("status").asText())
                        && payload.has("items") && payload.get("items").isArray()
                        && payload.get("items").size() > 0) {
                    providerResponseInfo.put("responseId", payload.get("responseId").asText());
                    providerResponseInfo.put("items", objectMapper.convertValue(payload.get("items"), List.class));
                } else {
                    log.warn("忽略无效的 provider_response 事件");
                }
                return Flux.empty();
            } else if ("usage".equals(type)) {
                // Handle usage data including latency
                JsonNode payload = data.path("payload");
                if (payload.has("latency_ms")) {
                    usageInfo.put("latency_ms", payload.get("latency_ms").asLong());
                }
                if (payload.has("completion_tokens")) {
                    usageInfo.put("completion_tokens", payload.get("completion_tokens").asInt());
                }
                if (payload.has("prompt_tokens")) {
                    usageInfo.put("prompt_tokens", payload.get("prompt_tokens").asInt());
                }
                if (payload.has("total_tokens")) {
                    usageInfo.put("total_tokens", payload.get("total_tokens").asInt());
                }
                if (payload.has("first_token_latency_ms") && !payload.get("first_token_latency_ms").isNull()) {
                    usageInfo.put("first_token_latency_ms", payload.get("first_token_latency_ms").asLong());
                }
                if (payload.has("is_success")) {
                    usageInfo.put("is_success", payload.get("is_success").asBoolean());
                }
                // Forward usage data to frontend
                return Flux.just(objectMapper.writeValueAsString(
                        Map.of("type", "usage", "payload", objectMapper.convertValue(payload, Map.class))
                ));
            }
        } catch (JsonProcessingException e) {
            log.error("解析SSE数据失败: {}", e.getMessage());
        }
        return Flux.empty();
    }

    private Mono<String> saveCompletedMessage(Long sessionId, Long userId, ChatStream chatStream,
                                               Long currentUserMessageId, StringBuffer sb, StringBuffer thinkingSb,
                                               List<Map<String, Object>> ragProcessList, ObjectMapper objectMapper,
                                               Map<String, Object> usageInfo, Map<String, Object> providerResponseInfo,
                                               AtomicBoolean saved) {
        return Mono.defer(() -> Mono.fromCallable(() -> {
            if (!saved.compareAndSet(false, true)) {
                return objectMapper.writeValueAsString(Map.of("type", "done"));
            }
            ConversationMessages aiMessage = new ConversationMessages();
            aiMessage.setSessionId(sessionId);
            aiMessage.setUserId(userId);
            aiMessage.setRole("assistant");
            aiMessage.setContent(sb.toString());
            // 保存思考内容
            if (thinkingSb.length() > 0) {
                aiMessage.setThinking(thinkingSb.toString());
            }
            aiMessage.setKbId(chatStream.getKbId());
            aiMessage.setStatus("completed");
            aiMessage.setModelId(chatStream.getModelId());
            if (providerResponseInfo.get("responseId") instanceof String
                    && providerResponseInfo.get("items") instanceof List) {
                aiMessage.setProviderResponseId((String) providerResponseInfo.get("responseId"));
                aiMessage.setProviderResponseItems(providerResponseInfo.get("items"));
            }
            // aiMessage.setCreatedAt(new Date());

            // Set latency if available
            if (usageInfo.containsKey("latency_ms")) {
                aiMessage.setLatencyMs((Long) usageInfo.get("latency_ms"));
            }
            // Set completion tokens if available
            if (usageInfo.containsKey("completion_tokens")) {
                aiMessage.setCompletionTokens((Integer) usageInfo.get("completion_tokens"));
            }
            // Set prompt tokens if available
            if (usageInfo.containsKey("prompt_tokens")) {
                aiMessage.setPromptTokens((Integer) usageInfo.get("prompt_tokens"));
            }
            // Set total tokens if available
            if (usageInfo.containsKey("total_tokens")) {
                aiMessage.setTotalTokens((Integer) usageInfo.get("total_tokens"));
            }
            // Set first token latency if available
            if (usageInfo.containsKey("first_token_latency_ms")) {
                aiMessage.setFirstTokenLatencyMs((Long) usageInfo.get("first_token_latency_ms"));
            }
            // Set success status if available
            if (usageInfo.containsKey("is_success")) {
                aiMessage.setIsSuccess((Boolean) usageInfo.get("is_success"));
            }


            if (!ragProcessList.isEmpty()) {
                try {
                    String ragContextJson = objectMapper.writeValueAsString(ragProcessList);
                    aiMessage.setRagContext(ragContextJson);
                } catch (JsonProcessingException e) {
                    log.error("序列化RAG过程信息失败", e);
                }
            }

            conversationMessagesService.saveRoundResult(currentUserMessageId, "completed", aiMessage);

            return objectMapper.writeValueAsString(Map.of(
                    "type", "done",
                    "userMessageId", currentUserMessageId,
                    "assistantMessageId", aiMessage.getId()
            ));
        }).subscribeOn(Schedulers.boundedElastic()));
    }

    private Mono<String> saveStoppedMessage(Long sessionId, Long userId, ChatStream chatStream,
                                            Long currentUserMessageId, StringBuffer sb, StringBuffer thinkingSb,
                                            List<Map<String, Object>> ragProcessList, ObjectMapper objectMapper, Map<String, Object> usageInfo,
                                            boolean clientCancelled) {
        return Mono.defer(() -> Mono.fromCallable(() -> {
            String messageStatus = clientCancelled ? "completed" : "error";
            ConversationMessages aiMessage = new ConversationMessages();
            aiMessage.setSessionId(sessionId);
            aiMessage.setUserId(userId);
            aiMessage.setRole("assistant");
            aiMessage.setContent(sb.toString());
            if (thinkingSb.length() > 0) {
                aiMessage.setThinking(thinkingSb.toString());
            }
            aiMessage.setKbId(chatStream.getKbId());
            aiMessage.setStatus(messageStatus);
            aiMessage.setModelId(chatStream.getModelId());

            if (usageInfo.containsKey("latency_ms")) {
                aiMessage.setLatencyMs((Long) usageInfo.get("latency_ms"));
            }
            if (usageInfo.containsKey("completion_tokens")) {
                aiMessage.setCompletionTokens((Integer) usageInfo.get("completion_tokens"));
            }
            if (usageInfo.containsKey("prompt_tokens")) {
                aiMessage.setPromptTokens((Integer) usageInfo.get("prompt_tokens"));
            }
            if (usageInfo.containsKey("total_tokens")) {
                aiMessage.setTotalTokens((Integer) usageInfo.get("total_tokens"));
            }
            if (usageInfo.containsKey("first_token_latency_ms")) {
                aiMessage.setFirstTokenLatencyMs((Long) usageInfo.get("first_token_latency_ms"));
            }
            if (usageInfo.containsKey("is_success")) {
                aiMessage.setIsSuccess((Boolean) usageInfo.get("is_success"));
            }

            if (!ragProcessList.isEmpty()) {
                try {
                    String ragContextJson = objectMapper.writeValueAsString(ragProcessList);
                    aiMessage.setRagContext(ragContextJson);
                } catch (JsonProcessingException e) {
                    log.error("序列化RAG过程信息失败", e);
                }
            }

            conversationMessagesService.saveRoundResult(currentUserMessageId, messageStatus, aiMessage);
            if (clientCancelled) {
                log.info("客户端中断，已保存部分内容为 assistant 消息，messageId={}, contentLength={}", aiMessage.getId(), sb.length());
            } else {
                log.error("流式请求异常，已保存部分内容为 error assistant 消息，messageId={}, contentLength={}", aiMessage.getId(), sb.length());
            }

            return "";
        }).subscribeOn(Schedulers.boundedElastic()));
    }

}
