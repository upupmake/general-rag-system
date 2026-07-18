package com.rag.ragserver.domain.openapi.vo;

import com.rag.ragserver.domain.Documents;
import lombok.Data;

@Data
public class OpenApiDocumentVO {
    private Long documentId;
    private String fileName;
    private Object status;

    public static OpenApiDocumentVO from(Documents document) {
        OpenApiDocumentVO result = new OpenApiDocumentVO();
        result.setDocumentId(document.getId());
        result.setFileName(document.getFileName());
        result.setStatus(document.getStatus());
        return result;
    }
}
