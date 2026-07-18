package com.rag.ragserver.service.impl;

import com.rag.ragserver.domain.KnowledgeBases;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseAccessVO;
import com.rag.ragserver.service.DocumentsService;
import com.rag.ragserver.service.KbPermissionService;
import com.rag.ragserver.service.KnowledgeBasesService;
import com.rag.ragserver.service.WorkspacesService;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
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
