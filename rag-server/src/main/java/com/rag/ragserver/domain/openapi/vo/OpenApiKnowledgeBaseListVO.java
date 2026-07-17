package com.rag.ragserver.domain.openapi.vo;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@Data
public class OpenApiKnowledgeBaseListVO {
    private List<OpenApiKnowledgeBaseVO> owned;
    private List<OpenApiWorkspaceKnowledgeBasesVO> workspaceShared;
    private List<OpenApiKnowledgeBaseVO> invited;
    @JsonProperty("public")
    private List<OpenApiKnowledgeBaseVO> publicKnowledgeBases;
}
