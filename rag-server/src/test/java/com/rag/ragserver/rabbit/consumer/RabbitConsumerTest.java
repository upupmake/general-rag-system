package com.rag.ragserver.rabbit.consumer;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.rag.ragserver.domain.Documents;
import com.rag.ragserver.rabbit.entity.DocumentProcessResponse;
import com.rag.ragserver.service.DocumentChunksService;
import com.rag.ragserver.service.DocumentsService;
import com.rag.ragserver.service.KnowledgeBasesService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

class RabbitConsumerTest {
    private DocumentsService documentsService;
    private RabbitConsumer consumer;

    @BeforeEach
    void setUp() {
        documentsService = mock(DocumentsService.class);
        consumer = new RabbitConsumer(
                documentsService,
                mock(DocumentChunksService.class),
                mock(KnowledgeBasesService.class),
                new ObjectMapper());
    }

    @Test
    void storesTheProcessorFailureReason() {
        Documents document = new Documents();
        document.setId(10L);
        when(documentsService.getById(10L)).thenReturn(document);
        DocumentProcessResponse response = DocumentProcessResponse.builder()
                .documentId(10L)
                .status("failed")
                .message("unsupported encoding")
                .build();

        consumer.receiveDocumentCompleteMessage(response);

        assertEquals("failed", document.getStatus());
        assertEquals("unsupported encoding", document.getErrorMessage());
        verify(documentsService).updateById(document);
    }

    @Test
    void clearsTheFailureReasonAfterSuccess() {
        Documents document = new Documents();
        document.setId(10L);
        document.setErrorMessage("old error");
        when(documentsService.getById(10L)).thenReturn(document);
        DocumentProcessResponse response = DocumentProcessResponse.builder()
                .documentId(10L)
                .status("success")
                .message("Document processed successfully")
                .build();

        consumer.receiveDocumentCompleteMessage(response);

        assertEquals("ready", document.getStatus());
        assertEquals("", document.getErrorMessage());
        verify(documentsService).updateById(document);
    }
}
