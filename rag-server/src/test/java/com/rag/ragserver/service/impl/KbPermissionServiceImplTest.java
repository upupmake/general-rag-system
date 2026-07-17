package com.rag.ragserver.service.impl;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.rag.ragserver.domain.KbShares;
import com.rag.ragserver.domain.KnowledgeBases;
import com.rag.ragserver.domain.WorkspaceMembers;
import com.rag.ragserver.service.DocumentsService;
import com.rag.ragserver.service.KbSharesService;
import com.rag.ragserver.service.KnowledgeBasesService;
import com.rag.ragserver.service.WorkspaceMembersService;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class KbPermissionServiceImplTest {
    @Test
    @SuppressWarnings("unchecked")
    void listReadableKbAccessSourcesUsesPriorityAndAllMemberWorkspaces() {
        KnowledgeBasesService knowledgeBasesService = mock(KnowledgeBasesService.class);
        KbSharesService kbSharesService = mock(KbSharesService.class);
        DocumentsService documentsService = mock(DocumentsService.class);
        WorkspaceMembersService workspaceMembersService = mock(WorkspaceMembersService.class);
        KbPermissionServiceImpl service = new KbPermissionServiceImpl(
                knowledgeBasesService,
                kbSharesService,
                documentsService,
                workspaceMembersService
        );

        WorkspaceMembers member = new WorkspaceMembers();
        member.setWorkspaceId(20L);
        when(workspaceMembersService.list(any(Wrapper.class)))
                .thenReturn(Collections.singletonList(member));

        KbShares invitedShared = new KbShares();
        invitedShared.setKbId(2L);
        KbShares invitedPublic = new KbShares();
        invitedPublic.setKbId(3L);
        when(kbSharesService.list(any(Wrapper.class)))
                .thenReturn(Arrays.asList(invitedShared, invitedPublic));

        KnowledgeBases ownedPublic = kb(1L, 7L, null, "public");
        KnowledgeBases shared = kb(2L, 8L, 20L, "shared");
        KnowledgeBases invitedPublicKb = kb(3L, 8L, null, "public");
        KnowledgeBases publicKb = kb(4L, 8L, null, "public");
        when(knowledgeBasesService.list(any(Wrapper.class)))
                .thenReturn(Collections.singletonList(ownedPublic))
                .thenReturn(Collections.singletonList(shared))
                .thenReturn(Arrays.asList(shared, invitedPublicKb))
                .thenReturn(Arrays.asList(ownedPublic, invitedPublicKb, publicKb));

        Map<Long, String> result = service.listReadableKbAccessSources(7L);

        assertEquals(4, result.size());
        assertEquals("owned", result.get(1L));
        assertEquals("workspace_shared", result.get(2L));
        assertEquals("invited", result.get(3L));
        assertEquals("public", result.get(4L));
    }

    private KnowledgeBases kb(Long id, Long ownerUserId, Long workspaceId, String visibility) {
        KnowledgeBases kb = new KnowledgeBases();
        kb.setId(id);
        kb.setOwnerUserId(ownerUserId);
        kb.setWorkspaceId(workspaceId);
        kb.setVisibility(visibility);
        return kb;
    }
}
