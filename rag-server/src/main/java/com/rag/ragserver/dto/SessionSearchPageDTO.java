package com.rag.ragserver.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class SessionSearchPageDTO {
    private List<SessionSearchResultDTO> items;
    private int nextOffset;
    private boolean hasMore;
}
