package com.rag.ragserver.controller;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.rag.ragserver.domain.ConversationMessages;
import com.rag.ragserver.domain.QuerySessions;
import com.rag.ragserver.domain.RequestLimitations;
import com.rag.ragserver.domain.model.vo.ModelPermission;
import com.rag.ragserver.dto.ChatStream;
import com.rag.ragserver.service.ConversationMessagesService;
import com.rag.ragserver.service.KbPermissionService;
import com.rag.ragserver.service.KnowledgeBasesService;
import com.rag.ragserver.service.QuerySessionsService;
import com.rag.ragserver.service.RequestLimitationsService;
import com.rag.ragserver.service.RolesService;
import com.rag.ragserver.utils.ChatStreamRegistry;
import com.rag.ragserver.utils.ModelUtils;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.http.HttpStatus;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import javax.servlet.http.HttpServletRequest;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

class ChatStreamLifecycleTest {
    private ConversationMessagesService messages;
    private RequestLimitationsService limitations;
    private ChatStreamRegistry registry;
    private AtomicInteger llmRequests;

    @BeforeEach
    void setUp() {
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(new MybatisConfiguration(), "chat-test");
        TableInfoHelper.initTableInfo(assistant, ConversationMessages.class);
        TableInfoHelper.initTableInfo(assistant, QuerySessions.class);
        TableInfoHelper.initTableInfo(assistant, RequestLimitations.class);
        messages = mock(ConversationMessagesService.class);
        limitations = mock(RequestLimitationsService.class);
        registry = mock(ChatStreamRegistry.class);
        llmRequests = new AtomicInteger();
        when(messages.saveRoundResult(any(), any(), any())).thenAnswer(invocation -> {
            ConversationMessages saved = invocation.getArgument(2);
            saved.setId(12L);
            return saved;
        });
        when(messages.update(any(Wrapper.class))).thenReturn(true);
    }

    @SuppressWarnings("unchecked")
    private Flux<String> stream(int delay, Mono<ClientResponse> response) {
        RequestLimitations limit = new RequestLimitations();
        limit.setDelaySecond(delay);
        when(limitations.getOne(any(Wrapper.class))).thenReturn(limit);
        WebClient client = WebClient.builder().exchangeFunction(request -> {
            llmRequests.incrementAndGet();
            return response;
        }).build();
        ChatController controller = new ChatController(
                mock(HttpServletRequest.class), mock(ModelUtils.class), mock(QuerySessionsService.class),
                messages, mock(KnowledgeBasesService.class), mock(KbPermissionService.class),
                mock(RolesService.class), limitations, client, registry);
        ChatStream chat = new ChatStream();
        chat.setSessionId(1L);
        chat.setModelId(7L);
        ConversationMessages user = new ConversationMessages();
        user.setId(11L);
        user.setRole("user");
        user.setContent("question");
        return ReflectionTestUtils.invokeMethod(controller, "executeStreamChat", chat, 2L,
                new ModelPermission(), Collections.singletonList(user), 11L);
    }

    @Test
    void cancellationDuringDelaySavesOneRoundWithoutCallingLlm() {
        StepVerifier.create(stream(30, Mono.never())).thenCancel().verify();

        assertEquals(0, llmRequests.get());
        ArgumentCaptor<ConversationMessages> saved = ArgumentCaptor.forClass(ConversationMessages.class);
        verify(messages, timeout(2000)).saveRoundResult(eq(11L), eq("completed"), saved.capture());
        assertEquals("assistant", saved.getValue().getRole());
        assertEquals("", saved.getValue().getContent());
        verify(registry, timeout(2000)).clear(1L);
    }

    @Test
    void cancellationBeforeFirstLlmEventSavesOneRound() {
        StepVerifier.create(stream(0, Mono.never())).thenCancel().verify();

        verify(messages, timeout(2000)).saveRoundResult(eq(11L), eq("completed"), any());
        verify(registry, timeout(2000)).clear(1L);
    }

    @Test
    void normalCompletionSavesIdsAndContentOnlyOnce() {
        ClientResponse response = ClientResponse.create(HttpStatus.OK)
                .header("Content-Type", "text/event-stream")
                .body("data: {\"type\":\"content\",\"payload\":\"\\\"answer\\\"\"}\n\n")
                .build();
        StepVerifier.create(stream(0, Mono.just(response)))
                .expectNextMatches(event -> event.contains("answer"))
                .expectNextMatches(event -> event.contains("assistantMessageId") && event.contains("12"))
                .verifyComplete();

        verify(messages, times(1)).saveRoundResult(eq(11L), eq("completed"), any());
        verify(registry, timeout(2000)).clear(1L);
    }

    @Test
    void llmFailurePersistsAnErrorRound() {
        StepVerifier.create(stream(0, Mono.error(new IllegalStateException("provider failure"))))
                .expectError().verify();

        verify(messages, timeout(2000)).saveRoundResult(eq(11L), eq("error"), any());
        verify(registry, timeout(2000)).clear(1L);
    }
}
