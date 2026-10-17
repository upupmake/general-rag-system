package com.rag.ragserver.rabbit.consumer;


import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.rag.ragserver.domain.DocumentChunks;
import com.rag.ragserver.domain.Documents;
import com.rag.ragserver.rabbit.entity.DocumentProcessResponse;
import com.rag.ragserver.service.DocumentChunksService;
import com.rag.ragserver.service.DocumentsService;
import com.rag.ragserver.service.KnowledgeBasesService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.annotation.Exchange;
import org.springframework.amqp.rabbit.annotation.Queue;
import org.springframework.amqp.rabbit.annotation.QueueBinding;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Component
@RequiredArgsConstructor
public class RabbitConsumer {
    private final DocumentsService documentsService;
    private final DocumentChunksService documentChunksService;
    private final KnowledgeBasesService knowledgeBasesService;
    private final ObjectMapper objectMapper;

    @RabbitListener(bindings = @QueueBinding(
            value = @Queue(value = "rag.document.complete.queue", durable = "true"),
            exchange = @Exchange(value = "server.interact.llm.exchange", type = "direct", durable = "true"),
            key = "rag.document.complete.key"
    ))
    public void receiveDocumentCompleteMessage(DocumentProcessResponse response) {
        // log.info("Received DocumentProcessResponse: {}", response);
        if ("success".equals(response.getStatus())) {
            Documents doc = documentsService.getById(response.getDocumentId());
            if (doc != null) {
                if (response.getChunks() == null || response.getChunks().isEmpty()) {
                    doc.setStatus("failed");
                    doc.setErrorMessage("文档未解析出任何有效内容");
                    documentsService.updateById(doc);
                    return;
                }
                doc.setStatus("ready");
                doc.setErrorMessage("");
                documentsService.updateById(doc);

                LambdaQueryWrapper<DocumentChunks> queryWrapper = new LambdaQueryWrapper<>();
                queryWrapper.eq(DocumentChunks::getDocumentId, response.getDocumentId());
                documentChunksService.remove(queryWrapper);

                List<DocumentChunks> chunks = response.getChunks().stream().map(dto -> {
                    DocumentChunks chunk = new DocumentChunks();
                    chunk.setDocumentId(response.getDocumentId());
                    chunk.setKbId(doc.getKbId());
                    chunk.setChunkIndex(dto.getChunkIndex());
                    chunk.setText(dto.getText());
                    chunk.setTokenLength(dto.getTokenLength());
                    chunk.setVectorId(dto.getVectorId());
                    try {
                        chunk.setMetadata(objectMapper.writeValueAsString(dto.getMetadata()));
                    } catch (Exception e) {
                        chunk.setMetadata("{}");
                    }
                    chunk.setCreatedAt(new Date());
                    return chunk;
                }).collect(Collectors.toList());
                documentChunksService.saveBatch(chunks);
            }
        } else {
            Documents doc = documentsService.getById(response.getDocumentId());
            if (doc != null) {
                doc.setStatus("failed");
                String reason = response.getMessage();
                if (reason == null || reason.trim().isEmpty()) {
                    reason = "文档处理失败";
                } else if (reason.length() > 1000) {
                    reason = reason.substring(0, 1000);
                }
                doc.setErrorMessage(reason);
                documentsService.updateById(doc);
            }
        }
    }
}
