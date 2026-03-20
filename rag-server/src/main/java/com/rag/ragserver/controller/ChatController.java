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
import java.util.Date;
import java.util.List;
import java.util.Map;

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
    private final WebClient webClient;

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
                .orderByAsc(ConversationMessages::getCreatedAt);
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

    private void updateMessageStatus(Long sessionId, Long messageId, String status) {
        Mono.fromRunnable(() -> {
            ConversationMessages message = conversationMessagesService.getById(messageId);
            if (message != null) {
                message.setStatus(status);
                conversationMessagesService.updateById(message);

                // 同时更新 Session 的最后活跃时间
                querySessionsService.update(
                        new LambdaUpdateWrapper<QuerySessions>()
                                .eq(QuerySessions::getId, sessionId)
                                .set(QuerySessions::getLastActiveAt, new Date())
                );
            } else {
                log.warn("未找到消息ID: {}", messageId);
            }
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
                        ConversationMessages::getOptions
                )
                .eq(ConversationMessages::getSessionId, sessionId)
                .eq(ConversationMessages::getUserId, userId)
                .and(w -> w.isNull(ConversationMessages::getIsDeleted).or().eq(ConversationMessages::getIsDeleted, 0))
                .orderByAsc(ConversationMessages::getCreatedAt);
        List<ConversationMessages> messageList = conversationMessagesService.list(messageQueryWrapper);
        if (messageList.isEmpty()) {
            throw new BusinessException(400, "会话不存在或无权限访问");
        }
        return messageList;
    }

    private Long processNewMessage(ChatStream chatStream, List<ConversationMessages> messageList,
                                   ConversationMessages lastMessage, Long userId) {
        Long currentUserMessageId = lastMessage.getId();
        String lastRole = (String) lastMessage.getRole();
        String lastStatus = (String) lastMessage.getStatus();

        if ("user".equals(lastRole) && "generating".equals(lastStatus)) {
            throw new BusinessException(400, "AI正在生成回复，请稍后再试");
        }
        if ("user".equals(lastRole) && "pending".equals(lastStatus)) {
            chatStream.setQuestion(lastMessage.getContent());
        } else if ("assistant".equals(lastRole)) {
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
                Map<String, Object> opts = chatStream.getOptions();
                if (opts.containsKey("thinking") && Boolean.FALSE.equals(opts.get("thinking"))) {
                    opts.remove("thinking");
                }
                newUserMessage.setOptions(opts);
            }

            conversationMessagesService.save(newUserMessage);
            messageList.add(newUserMessage);
            currentUserMessageId = newUserMessage.getId();
        } else {
            throw new BusinessException(500, "未知异常");
        }
        return currentUserMessageId;
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
        ObjectMapper objectMapper = new ObjectMapper();

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
        Map<String, Object> info = Map.of(
                "history", messageList,
                "model", modelPermission,
                "options", options
        );

        return webClient.post()
                .uri("/rag/chat/stream")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(info)
                .accept(MediaType.TEXT_EVENT_STREAM)
                .retrieve()
                .bodyToFlux(new ParameterizedTypeReference<ServerSentEvent<String>>() {
                })
                .doOnSubscribe(a -> updateMessageStatus(sessionId, currentUserMessageId, "generating"))
                .doOnError(e -> updateMessageStatus(sessionId, currentUserMessageId, "pending"))
                .doOnCancel(() -> updateMessageStatus(sessionId, currentUserMessageId, "pending"))
                .concatMap(event -> processStreamEvent(event, sb, thinkingSb, ragProcessList, objectMapper, usageInfo))
                .concatWith(saveCompletedMessage(sessionId, userId, chatStream, currentUserMessageId, sb, thinkingSb, ragProcessList, objectMapper, usageInfo));
    }

    private Flux<String> processStreamEvent(ServerSentEvent<String> event, StringBuffer sb, StringBuffer thinkingSb,
                                            List<Map<String, Object>> ragProcessList, ObjectMapper objectMapper, Map<String, Object> usageInfo) {
        String dataStr = event.data();
        if (dataStr == null || dataStr.isEmpty()) return Flux.empty();

        try {
            JsonNode data = objectMapper.readTree(dataStr);
            String type = data.path("type").asText();

            if ("content".equals(type) || "thinking".equals(type)) {
                String payloadJson = data.path("payload").asText();
                try {
                    String text = objectMapper.readValue(payloadJson, String.class);
                    if ("content".equals(type)) {
                        sb.append(text);
                    } else if ("thinking".equals(type)) {
                        thinkingSb.append(text);
                    }
                    return Flux.just(objectMapper.writeValueAsString(
                            Map.of("type", type, "content", text)
                    ));
                } catch (JsonProcessingException e) {
                    log.error("解析content payload失败: {}", e.getMessage());
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
                                              List<Map<String, Object>> ragProcessList, ObjectMapper objectMapper, Map<String, Object> usageInfo) {
        return Mono.defer(() -> Mono.fromCallable(() -> {
            ConversationMessages userMessage = conversationMessagesService.getById(currentUserMessageId);
            if (userMessage != null) {
                userMessage.setStatus("completed");
                conversationMessagesService.updateById(userMessage);
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

            if (!ragProcessList.isEmpty()) {
                try {
                    String ragContextJson = objectMapper.writeValueAsString(ragProcessList);
                    aiMessage.setRagContext(ragContextJson);
                } catch (JsonProcessingException e) {
                    log.error("序列化RAG过程信息失败", e);
                }
            }

            conversationMessagesService.save(aiMessage);

            return objectMapper.writeValueAsString(Map.of(
                    "type", "done",
                    "userMessageId", currentUserMessageId,
                    "assistantMessageId", aiMessage.getId()
            ));
        }).subscribeOn(Schedulers.boundedElastic()));
    }

}
