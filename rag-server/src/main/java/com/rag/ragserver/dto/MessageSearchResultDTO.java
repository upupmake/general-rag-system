package com.rag.ragserver.dto;

import lombok.Data;
import java.util.Date;

@Data
public class MessageSearchResultDTO {
    private Long messageId;
    private String content;
    private String matchType;
    private Long sessionId;
    private String sessionTitle;
    private Date createdAt;
}
