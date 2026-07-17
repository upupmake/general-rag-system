package com.rag.ragserver.controller;

import com.rag.ragserver.common.R;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseAccessVO;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseListVO;
import com.rag.ragserver.service.OpenApiKnowledgeBaseService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.servlet.http.HttpServletRequest;

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

    @GetMapping("/{kbId}/access")
    public R<OpenApiKnowledgeBaseAccessVO> checkAccess(@PathVariable Long kbId) {
        Long userId = (Long) request.getAttribute("userId");
        return R.success(openApiKnowledgeBaseService.checkAccess(kbId, userId));
    }
}
