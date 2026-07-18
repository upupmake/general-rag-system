package com.rag.ragserver.rabbit.consumer;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.rag.ragserver.domain.McpToolLogs;
import com.rag.ragserver.rabbit.entity.McpToolLogMessage;
import com.rag.ragserver.service.McpToolLogsService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.util.Date;

@Slf4j
@Component
@RequiredArgsConstructor
public class McpToolLogConsumer {
    private final McpToolLogsService mcpToolLogsService;
    private final ObjectMapper objectMapper;

    @RabbitListener(queues = "rag.mcp.tool.log.queue", containerFactory = "auditRabbitListenerContainerFactory")
    @Transactional(rollbackFor = Exception.class)
    public void receive(McpToolLogMessage message) {
        validate(message);

        McpToolLogs logRecord = new McpToolLogs();
        logRecord.setInvocationId(message.getInvocationId());
        logRecord.setUserId(message.getUserId());
        logRecord.setAccessKeyId(message.getAccessKeyId());
        logRecord.setToolName(message.getToolName());
        logRecord.setKnowledgeBaseId(message.getKnowledgeBaseId());
        logRecord.setDocumentId(message.getDocumentId());
        logRecord.setRequestSummary(toJson(message.getRequestSummary()));
        logRecord.setResultSummary(toJson(message.getResultSummary()));
        logRecord.setStatus(message.getStatus());
        logRecord.setErrorMessage(message.getErrorMessage());
        logRecord.setDurationMs(message.getDurationMs());
        logRecord.setCreatedAt(new Date(message.getCreatedAt()));

        mcpToolLogsService.saveIfAbsent(logRecord);
    }

    private void validate(McpToolLogMessage message) {
        if (message == null || message.getInvocationId() == null || message.getInvocationId().trim().isEmpty()
                || message.getUserId() == null || message.getAccessKeyId() == null
                || message.getToolName() == null || message.getToolName().trim().isEmpty()
                || message.getStatus() == null || message.getCreatedAt() == null) {
            throw new IllegalArgumentException("Invalid MCP tool log message");
        }
        if (!"SUCCESS".equals(message.getStatus()) && !"FAIL".equals(message.getStatus())) {
            throw new IllegalArgumentException("Invalid MCP tool log status");
        }
    }

    private String toJson(Object value) {
        if (value == null) {
            return null;
        }
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            throw new IllegalArgumentException("Invalid MCP tool log JSON", e);
        }
    }
}
