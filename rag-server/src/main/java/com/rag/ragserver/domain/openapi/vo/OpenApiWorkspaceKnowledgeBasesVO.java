package com.rag.ragserver.domain.openapi.vo;

import lombok.Data;

import java.util.List;

@Data
public class OpenApiWorkspaceKnowledgeBasesVO {
    private Long workspaceId;
    private String workspaceName;
    private List<OpenApiKnowledgeBaseVO> knowledgeBases;
}
