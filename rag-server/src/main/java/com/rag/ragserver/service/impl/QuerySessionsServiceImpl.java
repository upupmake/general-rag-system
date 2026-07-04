package com.rag.ragserver.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.rag.ragserver.assembler.SessionAssembler;
import com.rag.ragserver.domain.ConversationMessages;
import com.rag.ragserver.domain.QuerySessions;
import com.rag.ragserver.domain.session.vo.SessionListVO;
import com.rag.ragserver.dto.SessionCursorQuery;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.service.ConversationMessagesService;
import com.rag.ragserver.service.QuerySessionsService;
import com.rag.ragserver.mapper.QuerySessionsMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.text.SimpleDateFormat;
import java.util.*;

import com.rag.ragserver.dto.MessageSearchResultDTO;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.rag.ragserver.dto.SessionSearchResultDTO;

/**
 * @author make
 * @description 针对表【query_sessions(RAG 查询会话上下文表)】的数据库操作Service实现
 * @createDate 2026-01-02 22:22:17
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class QuerySessionsServiceImpl extends ServiceImpl<QuerySessionsMapper, QuerySessions>
        implements QuerySessionsService {
    private final WebClient webClient;
    private final ConversationMessagesService conversationMessagesService;
    private final QuerySessionsMapper querySessionsMapper;

    @Override
    public List<SessionSearchResultDTO> searchSessions(Long userId, Long workspaceId, String keyword, int limit, int offset) {
        // 1. 获取包含关键词的消息列表（分页针对消息）
        List<MessageSearchResultDTO> messages = querySessionsMapper.searchMessages(userId, workspaceId, keyword, limit, offset);
        
        if (messages.isEmpty()) {
            return Collections.emptyList();
        }

        // 2. 按 Session 分组并聚合，同时处理内容摘要
        Map<Long, SessionSearchResultDTO> sessionMap = new LinkedHashMap<>();
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        
        for (MessageSearchResultDTO msg : messages) {
            sessionMap.computeIfAbsent(msg.getSessionId(), k -> {
                SessionSearchResultDTO dto = new SessionSearchResultDTO();
                dto.setSessionId(msg.getSessionId());
                dto.setSessionTitle(msg.getSessionTitle());
                dto.setContentList(new ArrayList<>());
                return dto;
            });

            // 提取摘要：关键词前后约100字符
            List<String> extracted = extractSnippets(msg.getContent(), keyword);
            String timeStr = msg.getCreatedAt() != null ? sdf.format(msg.getCreatedAt()) : "";
            
            for (String content : extracted) {
                sessionMap.get(msg.getSessionId()).getContentList().add(
                    new SessionSearchResultDTO.Snippet(content, timeStr)
                );
            }
        }

        return new ArrayList<>(sessionMap.values());
    }

    private List<String> extractSnippets(String content, String keyword) {
        List<String> snippets = new ArrayList<>();
        if (content == null || keyword == null || keyword.isEmpty()) {
            return snippets;
        }
        int contextLen = 100; // 前后各100字

        String lowerContent = content.toLowerCase();
        String lowerKeyword = keyword.toLowerCase();
        int keywordLen = keyword.length();
        int contentLen = content.length();


        int index = 0;
        while ((index = lowerContent.indexOf(lowerKeyword, index)) != -1) {
            int start = Math.max(0, index - contextLen);
            int end = Math.min(contentLen, index + keywordLen + contextLen);

            String snippet = content.substring(start, end);
            
            // 如果不是从头开始，加省略号
            if (start > 0) snippet = "..." + snippet;
            // 如果不是到尾结束，加省略号
            if (end < contentLen) snippet = snippet + "...";
            
            snippets.add(snippet);
            
            // 移动索引，确保前进
            index = index + contextLen / 2;
            if (index >= contentLen) break;
        }
        return snippets;
    }

    @Override
    public String generateTitle(Long sessionId, Long userId, Long workspaceId) {
        QuerySessions session = getOne(new LambdaQueryWrapper<QuerySessions>()
                .eq(QuerySessions::getId, sessionId)
                .eq(QuerySessions::getUserId, userId)
                .eq(QuerySessions::getWorkspaceId, workspaceId));
        if (session == null) {
            throw new BusinessException(404, "会话不存在");
        }
        // 获取第一条用户消息
        ConversationMessages firstMsg = conversationMessagesService.getOne(
                new LambdaQueryWrapper<ConversationMessages>()
                        .eq(ConversationMessages::getSessionId, sessionId)
                        .eq(ConversationMessages::getRole, "user")
                        .orderByAsc(ConversationMessages::getCreatedAt)
                        .last("LIMIT 1")
        );
        String content = firstMsg != null ? firstMsg.getContent() : "";
        String title;
        try {
            Map<?, ?> response = webClient.post()
                    .uri("/rag/chat/session/name")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(Map.of("content", content))
                    .retrieve()
                    .bodyToMono(Map.class)
                    .block();
            title = response != null ? Objects.toString(response.get("title"), "新的对话") : "新的对话";
        } catch (Exception e) {
            log.error("Failed to generate session title for sessionId={}: {}", sessionId, e.getMessage());
            title = "新的对话";
        }
        update(new LambdaUpdateWrapper<QuerySessions>()
                .eq(QuerySessions::getId, sessionId)
                .set(QuerySessions::getSessionKey, title));
        return title;
    }

    @Override
    public SessionListVO listByCursor(Long userId, Long workspaceId, SessionCursorQuery query) {
        LambdaQueryWrapper<QuerySessions> qw = Wrappers.lambdaQuery();
        qw.eq(QuerySessions::getUserId, userId)
                .eq(QuerySessions::getWorkspaceId, workspaceId)
                .orderByDesc(QuerySessions::getLastActiveAt)
                .orderByDesc(QuerySessions::getId)
                .last("limit " + (query.getPageSize() + 1));

        if (query.getLastActiveAt() != null) {
            qw.and(w -> w
                    .lt(QuerySessions::getLastActiveAt, query.getLastActiveAt())
                    .or()
                    .eq(QuerySessions::getLastActiveAt, query.getLastActiveAt())
                    .lt(QuerySessions::getId, query.getLastId())
            );
        }
        List<QuerySessions> list = list(qw);

        boolean hasMore = list.size() > query.getPageSize();
        if (hasMore) list.remove(list.size() - 1);

        return SessionAssembler.toVO(list, hasMore);
    }

    @Override
    public Boolean deleteSession(Long sessionId, Long userId, Long workspaceId) {
        LambdaQueryWrapper<QuerySessions> qw = Wrappers.lambdaQuery();
        qw.eq(QuerySessions::getId, sessionId)
                .eq(QuerySessions::getUserId, userId)
                .eq(QuerySessions::getWorkspaceId, workspaceId)
                .eq(QuerySessions::getIsDeleted, 0);

        QuerySessions session = getOne(qw);
        if (session == null) {
            throw new BusinessException(404, "会话不存在或无权限删除");
        }
        return removeById(sessionId);
    }

    @Override
    public String renameSession(Long sessionId, Long userId, Long workspaceId, String title) {
        String trimmed = title == null ? "" : title.trim();
        if (trimmed.isEmpty()) {
            throw new BusinessException(400, "会话标题不能为空");
        }
        QuerySessions session = getOne(new LambdaQueryWrapper<QuerySessions>()
                .eq(QuerySessions::getId, sessionId)
                .eq(QuerySessions::getUserId, userId)
                .eq(QuerySessions::getWorkspaceId, workspaceId)
                .eq(QuerySessions::getIsDeleted, 0));
        if (session == null) {
            throw new BusinessException(404, "会话不存在或无权限修改");
        }
        update(new LambdaUpdateWrapper<QuerySessions>()
                .eq(QuerySessions::getId, sessionId)
                .set(QuerySessions::getSessionKey, trimmed));
        return trimmed;
    }
}




