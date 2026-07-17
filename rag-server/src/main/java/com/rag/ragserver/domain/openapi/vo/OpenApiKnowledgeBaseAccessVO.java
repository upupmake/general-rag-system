package com.rag.ragserver.domain.openapi.vo;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class OpenApiKnowledgeBaseAccessVO {
    private Long knowledgeBaseId;
    private boolean accessible;
    private String accessSource;
    private Long ownerUserId;
}
