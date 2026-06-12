package com.rag.ragserver.domain.model.vo;

import lombok.Data;

/**
 * 模型性能统计（models LEFT JOIN conversation_messages 聚合查询）
 */
@Data
public class ModelPerformanceStats {
    private Long modelId;
    private String modelName;
    private String provider;
    private Long requestCount;
    private Double avgLatency;
    private Double successRate;
}
