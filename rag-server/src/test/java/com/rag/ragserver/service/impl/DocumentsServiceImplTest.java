package com.rag.ragserver.service.impl;

import com.rag.ragserver.configuration.MinioConfig;
import com.rag.ragserver.domain.Documents;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.rabbit.entity.DocumentProcessMessage;
import com.rag.ragserver.service.DocumentChunksService;
import com.rag.ragserver.service.KnowledgeBasesService;
import io.milvus.v2.client.MilvusClientV2;
import io.minio.MinioClient;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.amqp.rabbit.core.RabbitTemplate;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

class DocumentsServiceImplTest {
    private MinioConfig minioConfig;
    private RabbitTemplate rabbitTemplate;
    private DocumentsServiceImpl service;

    @BeforeEach
    void setUp() {
        minioConfig = mock(MinioConfig.class);
        rabbitTemplate = mock(RabbitTemplate.class);
        when(minioConfig.getBucketName()).thenReturn("files");
        service = spy(new DocumentsServiceImpl(
                mock(MinioClient.class),
                minioConfig,
                mock(MilvusClientV2.class),
                rabbitTemplate,
                mock(KnowledgeBasesService.class),
                mock(DocumentChunksService.class)));
    }

    @Test
    void folderPathOfNormalizesPaths() {
        assertEquals("", DocumentsServiceImpl.folderPathOf(null));
        assertEquals("", DocumentsServiceImpl.folderPathOf("file.docx"));
        assertEquals("docs", DocumentsServiceImpl.folderPathOf("docs/file.docx"));
        assertEquals("docs/2026", DocumentsServiceImpl.folderPathOf("docs/2026/file.docx"));
        assertEquals("docs", DocumentsServiceImpl.folderPathOf("\\docs\\file.docx"));
        assertEquals("docs", DocumentsServiceImpl.folderPathOf("/docs//file.docx"));
        assertEquals("docs", DocumentsServiceImpl.folderPathOf("docs/file.docx/"));
    }

    @Test
    void findDuplicateOnlyMatchesSameFolder() {
        Documents root = documentNamed("paper.docx");
        Documents docs = documentNamed("docs/paper.docx");
        Documents archive = documentNamed("archive/paper.docx");
        List<Documents> candidates = Arrays.asList(root, archive, docs);

        assertSame(root, DocumentsServiceImpl.findDuplicateInFolder(candidates, ""));
        assertSame(docs, DocumentsServiceImpl.findDuplicateInFolder(candidates, "docs"));
        assertNull(DocumentsServiceImpl.findDuplicateInFolder(candidates, "other"));
        assertNull(DocumentsServiceImpl.findDuplicateInFolder(null, "docs"));
    }

    @Test
    void reprocessResetsStatusAndRepublishesExistingFile() {
        Documents document = document("failed");
        doReturn(document).when(service).getOne(any());
        doReturn(true).when(service).update(any());

        service.reprocessDocument(1L, 5L, 7L);

        verify(service).update(any());
        ArgumentCaptor<DocumentProcessMessage> captor = ArgumentCaptor.forClass(DocumentProcessMessage.class);
        verify(rabbitTemplate).convertAndSend(
                eq("server.interact.llm.exchange"), eq("rag.document.process.key"), captor.capture());
        DocumentProcessMessage message = captor.getValue();
        assertEquals(5L, message.getDocumentId());
        assertEquals(1L, message.getKbId());
        assertEquals(7L, message.getUserId());
        assertEquals("users/7/1/file.pdf", message.getFilePath());
        assertEquals("file.pdf", message.getFileName());
        assertEquals("files", message.getBucketName());
    }

    @Test
    void reprocessRejectsDocumentsThatAreNotFailed() {
        doReturn(document("ready")).when(service).getOne(any());

        BusinessException error = assertThrows(BusinessException.class,
                () -> service.reprocessDocument(1L, 5L, 7L));

        assertEquals(400, error.getCode());
        verifyNoInteractions(rabbitTemplate);
    }

    @Test
    void reprocessRejectsConcurrentClaim() {
        Documents document = document("failed");
        doReturn(document).when(service).getOne(any());
        doReturn(false).when(service).update(any());

        BusinessException error = assertThrows(BusinessException.class,
                () -> service.reprocessDocument(1L, 5L, 7L));

        assertEquals(400, error.getCode());
        verifyNoInteractions(rabbitTemplate);
    }

    @Test
    void reprocessReportsMissingDocument() {
        doReturn(null).when(service).getOne(any());

        BusinessException error = assertThrows(BusinessException.class,
                () -> service.reprocessDocument(1L, 5L, 7L));

        assertEquals(404, error.getCode());
    }

    private Documents document(String status) {
        Documents document = new Documents();
        document.setId(5L);
        document.setKbId(1L);
        document.setUploaderId(7L);
        document.setFilePath("users/7/1/file.pdf");
        document.setFileName("file.pdf");
        document.setStatus(status);
        document.setErrorMessage("old");
        return document;
    }

    private Documents documentNamed(String fileName) {
        Documents document = new Documents();
        document.setFileName(fileName);
        return document;
    }
}
