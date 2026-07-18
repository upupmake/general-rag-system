package com.rag.ragserver.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.rag.ragserver.common.R;
import com.rag.ragserver.domain.DocumentChunks;
import com.rag.ragserver.domain.Documents;
import com.rag.ragserver.domain.KnowledgeBases;
import com.rag.ragserver.domain.Users;
import com.rag.ragserver.domain.kb.vo.KbShareUserVO;
import com.rag.ragserver.dto.KbCreateDTO;
import com.rag.ragserver.dto.KbInviteDTO;
import com.rag.ragserver.domain.kb.KbVisibilityEnum;
import com.rag.ragserver.dto.KbUpdateDTO;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.service.DocumentChunksService;
import com.rag.ragserver.service.DocumentsService;
import com.rag.ragserver.service.KbPermissionService;
import com.rag.ragserver.service.KbSharesService;
import com.rag.ragserver.service.KnowledgeBasesService;
import com.rag.ragserver.service.UsersService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.validation.Valid;
import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/kb")
@RequiredArgsConstructor
public class KbController {
    private final HttpServletRequest request;
    private final KnowledgeBasesService kbService;
    private final DocumentsService documentsService;
    private final DocumentChunksService documentChunksService;
    private final KbPermissionService kbPermissionService;
    private final KbSharesService kbSharesService;
    private final UsersService usersService;

    @GetMapping
    public R<Map<String, List<KnowledgeBases>>> list() {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        return R.success(kbService.listByWorkspaceAndUser(workspaceId, userId));
    }

    @PostMapping
    public R<KnowledgeBases> createKnowledgeBase(@RequestBody @Validated KbCreateDTO kbCreateDTO) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        KnowledgeBases kb = new KnowledgeBases();
        kb.setName(kbCreateDTO.getName());
        kb.setDescription(kbCreateDTO.getDescription());
        kb.setOwnerUserId(userId);
        if ("shared".equals(kbCreateDTO.getVisibility().getValue())) {
            kb.setWorkspaceId(workspaceId);
        }
        kb.setVisibility(kbCreateDTO.getVisibility().name());
        KnowledgeBases created = kbService.createKnowledgeBase(kb);
        return R.success(created);
    }

    @DeleteMapping("/{kbId}")
    public R<Void> deleteKnowledgeBase(@PathVariable Long kbId) {
        Long userId = (Long) request.getAttribute("userId");
        kbService.deleteKnowledgeBase(kbId, userId);
        return R.success();
    }

    @PutMapping("/{kbId}")
    public R<Void> updateKnowledgeBase(@PathVariable Long kbId, @RequestBody KbUpdateDTO kbUpdateDTO) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        KnowledgeBases kb = kbService.getById(kbId);
        if (kb == null) {
            throw new BusinessException(404, "知识库不存在");
        }
        if (!kb.getOwnerUserId().equals(userId)) {
            throw new BusinessException(403, "没有权限修改知识库配置");
        }
        
        if (kbUpdateDTO.getName() != null) {
            kb.setName(kbUpdateDTO.getName().trim());
        }
        if (kbUpdateDTO.getDescription() != null) {
            kb.setDescription(kbUpdateDTO.getDescription().trim());
        }
        if (kbUpdateDTO.getSystemPrompt() != null) {
            kb.setSystemPrompt(kbUpdateDTO.getSystemPrompt().trim());
        }
        if (kbUpdateDTO.getVisibility() != null) {
            KbVisibilityEnum visibility = kbUpdateDTO.getVisibility();
            kb.setVisibility(visibility.name());
            if (visibility == KbVisibilityEnum.SHARED) {
                kb.setWorkspaceId(workspaceId);
            }
        }
        
        kbService.updateById(kb);

        // updateById 默认跳过 null 值；切换到 private/public 时需显式将 workspaceId 置空
        if (kbUpdateDTO.getVisibility() != null && kbUpdateDTO.getVisibility() != KbVisibilityEnum.SHARED) {
            kbService.update(new LambdaUpdateWrapper<KnowledgeBases>()
                    .eq(KnowledgeBases::getId, kbId)
                    .set(KnowledgeBases::getWorkspaceId, null));
        }

        return R.success();
    }

    @GetMapping("/{kbId}/documents")
    public R<List<Documents>> listDocuments(@PathVariable Long kbId) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        if (!kbPermissionService.canReadKb(kbId, userId, workspaceId)) {
            throw new BusinessException(403, "没有权限访问该知识库");
        }
        return R.success(documentsService.listByKbId(kbId));
    }

    @PostMapping("/{kbId}/documents")
    public R<Void> uploadDocuments(@PathVariable Long kbId, @RequestParam("files") MultipartFile[] files) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        if (!kbPermissionService.canWriteKb(kbId, userId, workspaceId)) {
            throw new BusinessException(403, "没有权限向该知识库上传文档");
        }
        documentsService.uploadDocuments(kbId, files, userId);
        return R.success();
    }

    @DeleteMapping("/{kbId}/documents/{docId}")
    public R<Void> deleteDocument(@PathVariable Long kbId, @PathVariable Long docId) {
        Long userId = (Long) request.getAttribute("userId");
        if (!kbPermissionService.canModifyDocument(docId, userId)) {
            throw new BusinessException(403, "没有权限删除该文档");
        }
        documentsService.deleteDocument(kbId, docId, userId);
        return R.success();
    }

    @GetMapping("/{kbId}/documents/{docId}/chunks")
    public R<IPage<DocumentChunks>> listChunks(@PathVariable Long kbId,
                                               @PathVariable Long docId,
                                               @RequestParam(defaultValue = "1") Integer page,
                                               @RequestParam(defaultValue = "10") Integer size) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        if (!kbPermissionService.canReadKb(kbId, userId, workspaceId)) {
            throw new BusinessException(403, "没有权限访问该知识库");
        }
        Page<DocumentChunks> chunkPage = new Page<>(page, size);
        LambdaQueryWrapper<DocumentChunks> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DocumentChunks::getDocumentId, docId);
        wrapper.eq(DocumentChunks::getKbId, kbId);
        wrapper.orderByAsc(DocumentChunks::getChunkIndex);
        return R.success(documentChunksService.page(chunkPage, wrapper));
    }

    @GetMapping("/{kbId}/documents/{docId}/preview")
    public void previewDocument(@PathVariable Long kbId, @PathVariable Long docId, HttpServletResponse response) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        if (!kbPermissionService.canReadKb(kbId, userId, workspaceId)) {
            throw new BusinessException(403, "没有权限访问该知识库");
        }
        response.setHeader("Cache-Control", "max-age=3600");
        documentsService.previewDocument(docId, userId, response);
    }

    /**
     * 邀请用户访问知识库
     */
    @PostMapping("/{kbId}/invite")
    public R<Void> inviteUserToKb(@PathVariable Long kbId, @Valid @RequestBody KbInviteDTO dto) {
        Long operatorUserId = (Long) request.getAttribute("userId");
        
        // 检查操作者是否是知识库拥有者
        if (!kbPermissionService.isKbOwner(kbId, operatorUserId)) {
            throw new BusinessException(403, "只有知识库拥有者可以邀请用户");
        }
        
        // 查找目标用户
        Users targetUser = usersService.lambdaQuery()
                .eq(Users::getUsername, dto.getUserIdentifier())
                .or()
                .eq(Users::getEmail, dto.getUserIdentifier())
                .one();
        
        if (targetUser == null) {
            throw new BusinessException(404, "目标用户不存在");
        }
        
        kbSharesService.inviteUser(kbId, targetUser.getId(), operatorUserId);
        
        return R.success();
    }

    /**
     * 获取知识库的被邀请用户列表
     */
    @GetMapping("/{kbId}/invited-users")
    public R<List<KbShareUserVO>> getInvitedUsers(@PathVariable Long kbId) {
        Long userId = (Long) request.getAttribute("userId");
        
        // 检查是否是知识库拥有者
        if (!kbPermissionService.isKbOwner(kbId, userId)) {
            throw new BusinessException(403, "只有知识库拥有者可以查看被邀请用户列表");
        }
        
        List<KbShareUserVO> invitedUsers = kbSharesService.getInvitedUsers(kbId);
        
        return R.success(invitedUsers);
    }

    /**
     * 移除被邀请用户
     */
    @DeleteMapping("/{kbId}/invited-users/{userId}")
    public R<Void> removeUserFromKb(@PathVariable Long kbId, @PathVariable Long userId) {
        Long operatorUserId = (Long) request.getAttribute("userId");
        
        // 检查是否是知识库拥有者
        if (!kbPermissionService.isKbOwner(kbId, operatorUserId)) {
            throw new BusinessException(403, "只有知识库拥有者可以移除被邀请用户");
        }
        
        kbSharesService.removeInvitedUser(kbId, userId, operatorUserId);
        
        return R.success();
    }
}
