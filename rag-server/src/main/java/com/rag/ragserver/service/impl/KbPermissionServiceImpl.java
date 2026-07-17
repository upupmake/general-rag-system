package com.rag.ragserver.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.rag.ragserver.domain.Documents;
import com.rag.ragserver.domain.KbShares;
import com.rag.ragserver.domain.KnowledgeBases;
import com.rag.ragserver.domain.WorkspaceMembers;
import com.rag.ragserver.service.DocumentsService;
import com.rag.ragserver.service.KbPermissionService;
import com.rag.ragserver.service.KbSharesService;
import com.rag.ragserver.service.KnowledgeBasesService;
import com.rag.ragserver.service.WorkspaceMembersService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class KbPermissionServiceImpl implements KbPermissionService {
    
    private final KnowledgeBasesService knowledgeBasesService;
    private final KbSharesService kbSharesService;
    private final DocumentsService documentsService;
    private final WorkspaceMembersService workspaceMembersService;
    
    @Override
    public boolean canReadKb(Long kbId, Long userId, Long workspaceId) {
        KnowledgeBases kb = knowledgeBasesService.getById(kbId);
        if (kb == null) {
            return false;
        }
        
        // 1. 如果是拥有者，可以访问
        if (userId.equals(kb.getOwnerUserId())) {
            return true;
        }
        
        // 2. 如果是public，所有人都可以读
        if ("public".equals(kb.getVisibility())) {
            return true;
        }
        
        // 3. 如果是shared，检查是否在同一workspace
        if ("shared".equals(kb.getVisibility()) && workspaceId != null 
            && workspaceId.equals(kb.getWorkspaceId())) {
            return true;
        }
        
        // 4. 检查是否在kb_shares中被授权
        LambdaQueryWrapper<KbShares> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(KbShares::getKbId, kbId)
               .eq(KbShares::getUserId, userId);
        long count = kbSharesService.count(wrapper);
        
        return count > 0;
    }

    @Override
    public String getReadAccessSource(Long kbId, Long userId) {
        KnowledgeBases kb = knowledgeBasesService.getById(kbId);
        if (kb == null) {
            return null;
        }
        if (userId.equals(kb.getOwnerUserId())) {
            return "owned";
        }
        if ("shared".equals(kb.getVisibility()) && kb.getWorkspaceId() != null) {
            LambdaQueryWrapper<WorkspaceMembers> memberWrapper = new LambdaQueryWrapper<>();
            memberWrapper.eq(WorkspaceMembers::getWorkspaceId, kb.getWorkspaceId())
                    .eq(WorkspaceMembers::getUserId, userId);
            if (workspaceMembersService.count(memberWrapper) > 0) {
                return "workspace_shared";
            }
        }
        LambdaQueryWrapper<KbShares> shareWrapper = new LambdaQueryWrapper<>();
        shareWrapper.eq(KbShares::getKbId, kbId)
                .eq(KbShares::getUserId, userId);
        if (kbSharesService.count(shareWrapper) > 0) {
            return "invited";
        }
        if ("public".equals(kb.getVisibility())) {
            return "public";
        }
        return null;
    }

    @Override
    public Map<Long, String> listReadableKbAccessSources(Long userId) {
        Map<Long, String> sources = new LinkedHashMap<>();

        LambdaQueryWrapper<KnowledgeBases> ownedWrapper = new LambdaQueryWrapper<>();
        ownedWrapper.eq(KnowledgeBases::getOwnerUserId, userId)
                .orderByDesc(KnowledgeBases::getCreatedAt);
        knowledgeBasesService.list(ownedWrapper)
                .forEach(kb -> sources.put(kb.getId(), "owned"));

        LambdaQueryWrapper<WorkspaceMembers> membersWrapper = new LambdaQueryWrapper<>();
        membersWrapper.select(WorkspaceMembers::getWorkspaceId)
                .eq(WorkspaceMembers::getUserId, userId);
        List<Long> workspaceIds = workspaceMembersService.list(membersWrapper).stream()
                .map(WorkspaceMembers::getWorkspaceId)
                .collect(Collectors.toList());
        if (!workspaceIds.isEmpty()) {
            LambdaQueryWrapper<KnowledgeBases> sharedWrapper = new LambdaQueryWrapper<>();
            sharedWrapper.eq(KnowledgeBases::getVisibility, "shared")
                    .in(KnowledgeBases::getWorkspaceId, workspaceIds)
                    .orderByDesc(KnowledgeBases::getCreatedAt);
            knowledgeBasesService.list(sharedWrapper)
                    .forEach(kb -> sources.putIfAbsent(kb.getId(), "workspace_shared"));
        }

        LambdaQueryWrapper<KbShares> sharesWrapper = new LambdaQueryWrapper<>();
        sharesWrapper.select(KbShares::getKbId)
                .eq(KbShares::getUserId, userId);
        List<Long> invitedKbIds = kbSharesService.list(sharesWrapper).stream()
                .map(KbShares::getKbId)
                .collect(Collectors.toList());
        if (!invitedKbIds.isEmpty()) {
            LambdaQueryWrapper<KnowledgeBases> invitedWrapper = new LambdaQueryWrapper<>();
            invitedWrapper.in(KnowledgeBases::getId, invitedKbIds)
                    .orderByDesc(KnowledgeBases::getCreatedAt);
            knowledgeBasesService.list(invitedWrapper)
                    .forEach(kb -> sources.putIfAbsent(kb.getId(), "invited"));
        }

        LambdaQueryWrapper<KnowledgeBases> publicWrapper = new LambdaQueryWrapper<>();
        publicWrapper.eq(KnowledgeBases::getVisibility, "public")
                .orderByDesc(KnowledgeBases::getCreatedAt);
        knowledgeBasesService.list(publicWrapper)
                .forEach(kb -> sources.putIfAbsent(kb.getId(), "public"));
        return Collections.unmodifiableMap(sources);
    }
    
    @Override
    public boolean canWriteKb(Long kbId, Long userId, Long workspaceId) {
        KnowledgeBases kb = knowledgeBasesService.getById(kbId);
        if (kb == null) {
            return false;
        }
        
        // 只有拥有者可以写入（上传文档等）
        // 即使是shared或public的知识库，也只有拥有者可以上传文档
        return userId.equals(kb.getOwnerUserId());
    }
    
    @Override
    public boolean isKbOwner(Long kbId, Long userId) {
        KnowledgeBases kb = knowledgeBasesService.getById(kbId);
        if (kb == null) {
            return false;
        }
        return userId.equals(kb.getOwnerUserId());
    }
    
    @Override
    public boolean canModifyDocument(Long docId, Long userId) {
        Documents document = documentsService.getById(docId);
        if (document == null) {
            return false;
        }
        
        // 1. 如果是文档上传者，可以修改
        if (userId.equals(document.getUploaderId())) {
            return true;
        }
        
        // 2. 如果是知识库拥有者，也可以修改
        return isKbOwner(document.getKbId(), userId);
    }
}
