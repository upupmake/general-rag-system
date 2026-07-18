package com.rag.ragserver.controller;

import com.rag.ragserver.common.R;
import com.rag.ragserver.domain.openapi.dto.OpenApiKnowledgeBaseCreateDTO;
import com.rag.ragserver.domain.openapi.vo.OpenApiDocumentVO;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseAccessVO;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseListVO;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseVO;
import com.rag.ragserver.service.OpenApiKnowledgeBaseService;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;
import java.util.List;

@RestController
@RequestMapping("/openapi/v1/knowledge-bases")
@RequiredArgsConstructor
public class OpenApiKnowledgeBasesController {
    private final OpenApiKnowledgeBaseService openApiKnowledgeBaseService;
    private final HttpServletRequest request;

    @GetMapping
    public R<OpenApiKnowledgeBaseListVO> list() {
        Long userId = (Long) request.getAttribute("userId");
        return R.success(openApiKnowledgeBaseService.listReadableKnowledgeBases(userId));
    }

    @PostMapping
    public R<OpenApiKnowledgeBaseVO> create(
            @RequestBody @Validated OpenApiKnowledgeBaseCreateDTO createDTO) {
        Long userId = (Long) request.getAttribute("userId");
        return R.success(openApiKnowledgeBaseService.createKnowledgeBase(createDTO, userId));
    }

    @GetMapping("/{kbId}/access")
    public R<OpenApiKnowledgeBaseAccessVO> checkAccess(@PathVariable Long kbId) {
        Long userId = (Long) request.getAttribute("userId");
        return R.success(openApiKnowledgeBaseService.checkAccess(kbId, userId));
    }

    @GetMapping("/{kbId}/private-access")
    public R<OpenApiKnowledgeBaseAccessVO> checkPrivateAccess(@PathVariable Long kbId) {
        Long userId = (Long) request.getAttribute("userId");
        return R.success(openApiKnowledgeBaseService.checkPrivateAccess(kbId, userId));
    }

    @PostMapping("/{kbId}/documents")
    public R<List<OpenApiDocumentVO>> uploadDocuments(
            @PathVariable Long kbId,
            @RequestParam("files") MultipartFile[] files) {
        Long userId = (Long) request.getAttribute("userId");
        return R.success(openApiKnowledgeBaseService.uploadPrivateDocuments(kbId, files, userId));
    }

    @DeleteMapping("/{kbId}/documents/{docId}")
    public R<Void> deleteDocument(@PathVariable Long kbId, @PathVariable Long docId) {
        Long userId = (Long) request.getAttribute("userId");
        openApiKnowledgeBaseService.deletePrivateDocument(kbId, docId, userId);
        return R.success();
    }
}
