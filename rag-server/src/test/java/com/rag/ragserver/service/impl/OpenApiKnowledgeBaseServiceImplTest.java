package com.rag.ragserver.service.impl;

import com.rag.ragserver.domain.KnowledgeBases;
import com.rag.ragserver.domain.openapi.dto.OpenApiKnowledgeBaseCreateDTO;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseAccessVO;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseVO;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.service.DocumentsService;
import com.rag.ragserver.service.KbPermissionService;
import com.rag.ragserver.service.KnowledgeBasesService;
import com.rag.ragserver.service.WorkspacesService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OpenApiKnowledgeBaseServiceImplTest {
    private final KbPermissionService permissionService = mock(KbPermissionService.class);
    private final KnowledgeBasesService knowledgeBasesService = mock(KnowledgeBasesService.class);
    private final OpenApiKnowledgeBaseServiceImpl service = new OpenApiKnowledgeBaseServiceImpl(
            permissionService,
            knowledgeBasesService,
            mock(WorkspacesService.class),
            mock(DocumentsService.class)
    );

    @ParameterizedTest
    @ValueSource(strings = {"private", "public"})
    void createKnowledgeBaseCreatesOwnedKnowledgeBaseWithoutWorkspace(String visibility) {
        OpenApiKnowledgeBaseCreateDTO createDTO = new OpenApiKnowledgeBaseCreateDTO();
        createDTO.setName("API 文档");
        createDTO.setDescription("知识库说明");
        createDTO.setVisibility(visibility);
        when(knowledgeBasesService.createKnowledgeBase(any(KnowledgeBases.class)))
                .thenAnswer(invocation -> {
                    KnowledgeBases knowledgeBase = invocation.getArgument(0);
                    knowledgeBase.setId(123L);
                    return knowledgeBase;
                });

        OpenApiKnowledgeBaseVO result = service.createKnowledgeBase(createDTO, 7L);

        ArgumentCaptor<KnowledgeBases> captor = ArgumentCaptor.forClass(KnowledgeBases.class);
        verify(knowledgeBasesService).createKnowledgeBase(captor.capture());
        KnowledgeBases created = captor.getValue();
        assertEquals("API 文档", created.getName());
        assertEquals("知识库说明", created.getDescription());
        assertEquals(7L, created.getOwnerUserId());
        assertEquals(visibility, created.getVisibility());
        assertNull(created.getWorkspaceId());
        assertEquals(123L, result.getId());
        assertEquals("owned", result.getAccessSource());
    }

    @Test
    void createKnowledgeBaseRejectsSharedVisibility() {
        OpenApiKnowledgeBaseCreateDTO createDTO = new OpenApiKnowledgeBaseCreateDTO();
        createDTO.setName("工作空间知识库");
        createDTO.setVisibility("shared");

        BusinessException exception = assertThrows(
                BusinessException.class,
                () -> service.createKnowledgeBase(createDTO, 7L)
        );

        assertEquals(400, exception.getCode());
        assertEquals("目前仅支持创建私有或公开知识库", exception.getMessage());
    }

    @Test
    void checkAccessReturnsOwnerOnlyWhenReadable() {
        KnowledgeBases knowledgeBase = new KnowledgeBases();
        knowledgeBase.setId(123L);
        knowledgeBase.setOwnerUserId(42L);
        when(permissionService.getReadAccessSource(123L, 7L)).thenReturn("invited");
        when(knowledgeBasesService.getById(123L)).thenReturn(knowledgeBase);

        OpenApiKnowledgeBaseAccessVO result = service.checkAccess(123L, 7L);

        assertTrue(result.isAccessible());
        assertEquals("invited", result.getAccessSource());
        assertEquals(42L, result.getOwnerUserId());
    }

    @Test
    void privateAccessRequiresOwnerAndPrivateVisibility() {
        KnowledgeBases knowledgeBase = new KnowledgeBases();
        knowledgeBase.setId(123L);
        knowledgeBase.setOwnerUserId(7L);
        knowledgeBase.setVisibility("private");
        when(knowledgeBasesService.getById(123L)).thenReturn(knowledgeBase);

        OpenApiKnowledgeBaseAccessVO result = service.checkPrivateAccess(123L, 7L);

        assertTrue(result.isAccessible());
        assertEquals("owned_private", result.getAccessSource());
        assertEquals(7L, result.getOwnerUserId());
    }

    @Test
    void checkAccessDoesNotExposeOwnerWhenUnreadable() {
        when(permissionService.getReadAccessSource(123L, 7L)).thenReturn(null);

        OpenApiKnowledgeBaseAccessVO result = service.checkAccess(123L, 7L);

        assertFalse(result.isAccessible());
        assertNull(result.getAccessSource());
        assertNull(result.getOwnerUserId());
    }
}
