package com.rag.ragserver.service;

import com.rag.ragserver.domain.openapi.dto.OpenApiKnowledgeBaseCreateDTO;
import com.rag.ragserver.domain.openapi.vo.OpenApiDocumentVO;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseAccessVO;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseListVO;
import com.rag.ragserver.domain.openapi.vo.OpenApiKnowledgeBaseVO;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

public interface OpenApiKnowledgeBaseService {
    OpenApiKnowledgeBaseListVO listReadableKnowledgeBases(Long userId);

    OpenApiKnowledgeBaseVO createKnowledgeBase(OpenApiKnowledgeBaseCreateDTO createDTO, Long userId);

    OpenApiKnowledgeBaseAccessVO checkAccess(Long kbId, Long userId);

    OpenApiKnowledgeBaseAccessVO checkPrivateAccess(Long kbId, Long userId);

    List<OpenApiDocumentVO> uploadPrivateDocuments(Long kbId, MultipartFile[] files, Long userId);

    void deletePrivateDocument(Long kbId, Long docId, Long userId);
}
