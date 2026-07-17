package com.rag.ragserver.service.impl;

import com.rag.ragserver.domain.KnowledgeBases;
import com.rag.ragserver.domain.Workspaces;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseAccessVO;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseListVO;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseVO;
import com.rag.ragserver.domain.openapi.vo.OpenApiWorkspaceKnowledgeBasesVO;
import com.rag.ragserver.service.KbPermissionService;
import com.rag.ragserver.service.KnowledgeBasesService;
import com.rag.ragserver.service.OpenApiKnowledgeBaseService;
import com.rag.ragserver.service.WorkspacesService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class OpenApiKnowledgeBaseServiceImpl implements OpenApiKnowledgeBaseService {
    private final KbPermissionService kbPermissionService;
    private final KnowledgeBasesService knowledgeBasesService;
    private final WorkspacesService workspacesService;

    @Override
    public OpenApiKnowledgeBaseListVO listReadableKnowledgeBases(Long userId) {
        Map<Long, String> accessSources = kbPermissionService.listReadableKbAccessSources(userId);
        List<KnowledgeBases> knowledgeBases = accessSources.isEmpty()
                ? Collections.emptyList()
                : knowledgeBasesService.lambdaQuery()
                        .in(KnowledgeBases::getId, accessSources.keySet())
                        .orderByDesc(KnowledgeBases::getCreatedAt)
                        .list();
        Map<Long, KnowledgeBases> knowledgeBaseMap = knowledgeBases.stream()
                .collect(Collectors.toMap(KnowledgeBases::getId, Function.identity()));

        Set<Long> workspaceIds = knowledgeBases.stream()
                .map(KnowledgeBases::getWorkspaceId)
                .filter(java.util.Objects::nonNull)
                .collect(Collectors.toSet());
        Map<Long, Workspaces> workspaceMap = workspaceIds.isEmpty()
                ? Collections.emptyMap()
                : workspacesService.listByIds(workspaceIds).stream()
                        .collect(Collectors.toMap(Workspaces::getId, Function.identity()));

        List<OpenApiKnowledgeBaseVO> owned = new ArrayList<>();
        List<OpenApiKnowledgeBaseVO> invited = new ArrayList<>();
        List<OpenApiKnowledgeBaseVO> publicKnowledgeBases = new ArrayList<>();
        Map<Long, OpenApiWorkspaceKnowledgeBasesVO> workspaceGroups = new LinkedHashMap<>();

        accessSources.forEach((kbId, accessSource) -> {
            KnowledgeBases kb = knowledgeBaseMap.get(kbId);
            if (kb == null) {
                return;
            }
            Workspaces workspace = kb.getWorkspaceId() == null ? null : workspaceMap.get(kb.getWorkspaceId());
            OpenApiKnowledgeBaseVO vo = toVO(kb, accessSource, workspace);
            if ("owned".equals(accessSource)) {
                owned.add(vo);
            } else if ("workspace_shared".equals(accessSource)) {
                OpenApiWorkspaceKnowledgeBasesVO group = workspaceGroups.computeIfAbsent(kb.getWorkspaceId(), id -> {
                    OpenApiWorkspaceKnowledgeBasesVO item = new OpenApiWorkspaceKnowledgeBasesVO();
                    item.setWorkspaceId(id);
                    item.setWorkspaceName(workspace == null ? null : workspace.getName());
                    item.setKnowledgeBases(new ArrayList<>());
                    return item;
                });
                group.getKnowledgeBases().add(vo);
            } else if ("invited".equals(accessSource)) {
                invited.add(vo);
            } else if ("public".equals(accessSource)) {
                publicKnowledgeBases.add(vo);
            }
        });

        OpenApiKnowledgeBaseListVO result = new OpenApiKnowledgeBaseListVO();
        result.setOwned(owned);
        result.setWorkspaceShared(new ArrayList<>(workspaceGroups.values()));
        result.setInvited(invited);
        result.setPublicKnowledgeBases(publicKnowledgeBases);
        return result;
    }

    @Override
    public OpenApiKnowledgeBaseAccessVO checkAccess(Long kbId, Long userId) {
        String accessSource = kbPermissionService.getReadAccessSource(kbId, userId);
        KnowledgeBases kb = accessSource == null ? null : knowledgeBasesService.getById(kbId);
        return new OpenApiKnowledgeBaseAccessVO(
                kbId,
                accessSource != null,
                accessSource,
                kb == null ? null : kb.getOwnerUserId()
        );
    }

    private OpenApiKnowledgeBaseVO toVO(KnowledgeBases kb, String accessSource, Workspaces workspace) {
        OpenApiKnowledgeBaseVO result = new OpenApiKnowledgeBaseVO();
        result.setId(kb.getId());
        result.setName(kb.getName());
        result.setDescription(kb.getDescription());
        result.setVisibility(String.valueOf(kb.getVisibility()));
        result.setWorkspaceId(kb.getWorkspaceId());
        result.setWorkspaceName(workspace == null ? null : workspace.getName());
        result.setAccessSource(accessSource);
        result.setCreatedAt(kb.getCreatedAt());
        result.setUpdatedAt(kb.getUpdatedAt());
        return result;
    }
}
