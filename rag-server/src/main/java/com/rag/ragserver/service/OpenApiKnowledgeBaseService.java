package com.rag.ragserver.service;

import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseAccessVO;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseListVO;

public interface OpenApiKnowledgeBaseService {
    OpenApiKnowledgeBaseListVO listReadableKnowledgeBases(Long userId);

    OpenApiKnowledgeBaseAccessVO checkAccess(Long kbId, Long userId);
}
