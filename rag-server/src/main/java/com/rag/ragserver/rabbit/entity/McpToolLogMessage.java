package com.rag.ragserver.rabbit.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class McpToolLogMessage implements Serializable {
    private String invocationId;
    private Long userId;
    private Long accessKeyId;
    private String toolName;
    private Long knowledgeBaseId;
    private Long documentId;
    private Map<String, Object> requestSummary;
    private Map<String, Object> resultSummary;
    private String status;
    private String errorMessage;
    private Long durationMs;
    private Long createdAt;
}
