package com.rag.ragserver.dto;

import lombok.Data;

import javax.validation.constraints.NotNull;
import java.time.LocalDateTime;

@Data
public class SessionCursorQuery {
    private LocalDateTime lastActiveAt;
    private Long lastId;
    private Integer pageSize = 20;
    private String keyword;
}
