package com.rag.ragserver.domain;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.util.Date;

@Data
@TableName("mcp_tool_logs")
public class McpToolLogs {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String invocationId;
    private Long userId;
    private Long accessKeyId;
    private String toolName;
    private Long knowledgeBaseId;
    private Long documentId;
    private String requestSummary;
    private String resultSummary;
    private String status;
    private String errorMessage;
    private Long durationMs;
    private Date createdAt;
}
