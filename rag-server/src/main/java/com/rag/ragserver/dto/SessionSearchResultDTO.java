package com.rag.ragserver.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

@Data
public class SessionSearchResultDTO {
    private Long sessionId;
    private String sessionTitle;
    private List<Snippet> contentList;

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Snippet {
        private String content;
        private String createdAt;
        private String matchType;
    }
}
