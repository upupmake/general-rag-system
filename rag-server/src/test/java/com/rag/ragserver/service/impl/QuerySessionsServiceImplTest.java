package com.rag.ragserver.service.impl;

import com.rag.ragserver.dto.MessageSearchResultDTO;
import com.rag.ragserver.dto.SessionSearchPageDTO;
import com.rag.ragserver.mapper.QuerySessionsMapper;
import com.rag.ragserver.service.ConversationMessagesService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Arrays;
import java.util.Collections;
import java.util.Date;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class QuerySessionsServiceImplTest {
    private QuerySessionsMapper mapper;
    private QuerySessionsServiceImpl service;

    @BeforeEach
    void setUp() {
        mapper = mock(QuerySessionsMapper.class);
        service = new QuerySessionsServiceImpl(
                mock(WebClient.class),
                mock(ConversationMessagesService.class),
                mapper);
    }

    @Test
    void advancesOffsetByConsumedHitRowsInsteadOfGroupedSessions() {
        MessageSearchResultDTO first = message(1L, 100L, "needle one", "content");
        MessageSearchResultDTO second = message(1L, 101L, "needle two", "content");
        MessageSearchResultDTO third = message(2L, 102L, "needle three", "content");
        when(mapper.searchMessages(1L, 2L, "needle", 3, 40))
                .thenReturn(Arrays.asList(first, second, third));

        SessionSearchPageDTO page = service.searchSessions(1L, 2L, "needle", 2, 40);

        assertEquals(1, page.getItems().size());
        assertEquals(2, page.getItems().get(0).getContentList().size());
        assertEquals(42, page.getNextOffset());
        assertTrue(page.isHasMore());
    }

    @Test
    void returnsTitleSnippetWithTitleMatchType() {
        MessageSearchResultDTO titleHit = message(7L, null, "RAG 入门指南", "title");
        when(mapper.searchMessages(1L, 2L, "RAG", 21, 0))
                .thenReturn(Collections.singletonList(titleHit));

        SessionSearchPageDTO page = service.searchSessions(1L, 2L, "RAG", 20, 0);

        assertEquals(1, page.getItems().size());
        assertEquals("title", page.getItems().get(0).getContentList().get(0).getMatchType());
        assertEquals(1, page.getNextOffset());
        assertFalse(page.isHasMore());
    }

    @Test
    void emptyResultKeepsOffsetAndStopsPaging() {
        when(mapper.searchMessages(1L, 2L, "none", 21, 30))
                .thenReturn(Collections.emptyList());

        SessionSearchPageDTO page = service.searchSessions(1L, 2L, "none", 20, 30);

        assertTrue(page.getItems().isEmpty());
        assertEquals(30, page.getNextOffset());
        assertFalse(page.isHasMore());
    }

    private MessageSearchResultDTO message(Long sessionId, Long messageId, String content, String matchType) {
        MessageSearchResultDTO dto = new MessageSearchResultDTO();
        dto.setSessionId(sessionId);
        dto.setMessageId(messageId);
        dto.setContent(content);
        dto.setSessionTitle("会话 " + sessionId);
        dto.setMatchType(matchType);
        dto.setCreatedAt(new Date());
        return dto;
    }
}
