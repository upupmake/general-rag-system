package com.rag.ragserver.domain.model.vo;

import lombok.Data;

@Data
public class ModelPerformanceVO {
    private Long modelId;
    private String modelName;
    private String provider;
    private Double successRate;
    private Long avgFirstTokenLatencyMs;
    private Long requestCount;
}
