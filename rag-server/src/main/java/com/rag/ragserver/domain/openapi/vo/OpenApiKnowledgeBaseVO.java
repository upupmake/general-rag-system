package com.rag.ragserver.domain.openapi.vo;

import lombok.Data;

import java.util.Date;

@Data
public class OpenApiKnowledgeBaseVO {
    private Long id;
    private String name;
    private String description;
    private String visibility;
    private Long workspaceId;
    private String workspaceName;
    private String accessSource;
    private Date createdAt;
    private Date updatedAt;
}
