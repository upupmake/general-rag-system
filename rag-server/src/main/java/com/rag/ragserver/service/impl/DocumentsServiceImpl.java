package com.rag.ragserver.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.rag.ragserver.configuration.MinioConfig;
import com.rag.ragserver.configuration.MilvusConfig;
import com.rag.ragserver.domain.Documents;
import com.rag.ragserver.domain.KnowledgeBases;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.mapper.DocumentsMapper;
import com.rag.ragserver.rabbit.entity.DocumentProcessMessage;
import com.rag.ragserver.service.DocumentChunksService;
import com.rag.ragserver.service.DocumentsService;
import com.rag.ragserver.service.KnowledgeBasesService;
import io.milvus.v2.service.collection.request.GetLoadStateReq;
import io.milvus.v2.service.collection.request.LoadCollectionReq;
import io.milvus.v2.service.collection.request.ReleaseCollectionReq;
import io.minio.*;
import io.minio.http.Method;
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.vector.request.DeleteReq;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.DigestUtils;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import javax.annotation.PostConstruct;
import javax.servlet.http.HttpServletResponse;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

/**
 * @author make
 * @description 针对表【documents(知识库原始文档表)】的数据库操作Service实现
 * @createDate 2025-12-31 01:13:35
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DocumentsServiceImpl extends ServiceImpl<DocumentsMapper, Documents> implements DocumentsService {
    private final MinioClient minioClient;
    private final MinioConfig minioConfig;
    private final MilvusClientV2 milvusClientV2;
    private final RabbitTemplate rabbitTemplate;
    private final KnowledgeBasesService knowledgeBasesService;
    private final DocumentChunksService documentChunksService;

    @PostConstruct
    public void init() {
        try {
            boolean found = minioClient.bucketExists(BucketExistsArgs.builder().bucket(minioConfig.getBucketName()).build());
            if (!found) {
                minioClient.makeBucket(MakeBucketArgs.builder().bucket(minioConfig.getBucketName()).build());
                log.info("Bucket '{}' created.", minioConfig.getBucketName());
            } else {
                log.info("Bucket '{}' already exists.", minioConfig.getBucketName());
            }
        } catch (Exception e) {
            log.error("Error checking/creating bucket: {}", e.getMessage());
        }
    }

    @Override
    public List<Documents> listByKbId(Long kbId) {
        LambdaQueryWrapper<Documents> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(Documents::getKbId, kbId).orderByDesc(Documents::getCreatedAt);
        return this.list(queryWrapper);
    }

    @Override
    public Long countActiveByUploaderId(Long uploaderId) {
        return this.baseMapper.countActiveByUploaderId(uploaderId);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public List<Documents> uploadDocuments(Long kbId, MultipartFile[] files, Long userId) {
        LambdaQueryWrapper<KnowledgeBases> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(KnowledgeBases::getId, kbId)
                .eq(KnowledgeBases::getOwnerUserId, userId);
        KnowledgeBases kb = knowledgeBasesService.getOne(queryWrapper);
        if (kb == null) {
            throw new BusinessException(403, "没有权限操作该知识库");
        }
        List<Documents> uploadedDocuments = new java.util.ArrayList<>();
        for (MultipartFile file : files) {
            try {
                String originalFilename = file.getOriginalFilename();
                if (originalFilename != null) {
                    originalFilename = originalFilename.replace("\\", "/");
                }
                String extension = "";
                if (originalFilename != null && originalFilename.contains(".")) {
                    extension = originalFilename.substring(originalFilename.lastIndexOf("."));
                }

                // 文件名中的空格自动替换为下划线
                if (originalFilename != null && originalFilename.contains(" ")) {
                    originalFilename = originalFilename.replace(" ", "_");
                    log.info("文件名包含空格，已自动替换为下划线: {}", originalFilename);
                }

                // File organization: users/{groupId}/{kbId}/{uuid}{ext}
                long groupId = userId % 1000;
                String objectName = String.format("users/%d/%d/%s%s", groupId, kbId, UUID.randomUUID().toString(), extension);

                String contentType = file.getContentType();

                long size = file.getSize();

                // Calculate checksum
                String checksum;
                try (InputStream is = file.getInputStream()) {
                    checksum = DigestUtils.md5DigestAsHex(is);
                }

                String folderPath = folderPathOf(originalFilename);
                // 锁定同一知识库 + 内容范围的索引间隙，避免并发上传绕过目录级去重
                List<Documents> candidates = this.list(new LambdaQueryWrapper<Documents>()
                        .select(Documents::getId, Documents::getFileName)
                        .eq(Documents::getKbId, kbId)
                        .eq(Documents::getChecksum, checksum)
                        .last("FOR UPDATE"));
                Documents duplicate = findDuplicateInFolder(candidates, folderPath);

                if (duplicate != null) {
                    throw new BusinessException(400, "同一目录下文件内容已存在: " + duplicate.getFileName());
                }

                try (InputStream is = file.getInputStream()) {
                    minioClient.putObject(PutObjectArgs.builder().bucket(minioConfig.getBucketName()).object(objectName).contentType(contentType).stream(is, size, -1).build());
                }

                Documents document = new Documents();
                document.setKbId(kbId);
                document.setFilePath(objectName);
                document.setFileName(originalFilename);
                document.setMimeType(contentType);
                document.setFileSize(size);
                document.setUploaderId(userId);
                document.setStatus("processing");
                document.setChecksum(checksum);
                // document.setCreatedAt(new Date());
                // document.setUpdatedAt(new Date());
                this.save(document);
                uploadedDocuments.add(document);
                DocumentProcessMessage message = DocumentProcessMessage.builder().documentId(document.getId()).kbId(kbId).userId(userId).filePath(objectName).fileName(originalFilename).bucketName(minioConfig.getBucketName()).build();
                rabbitTemplate.convertAndSend("server.interact.llm.exchange", "rag.document.process.key", message, msg -> {
                    log.info("Document processing message sent: {}", msg);
                    return msg;
                });

            } catch (Exception e) {
                if (e instanceof BusinessException) {
                    throw (BusinessException) e;
                }
                log.error("File upload failed: {}", file.getOriginalFilename(), e);
                throw new BusinessException(500, "文件上传失败: " + file.getOriginalFilename());
            }
        }
        return uploadedDocuments;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteDocument(Long kbId, Long docId, Long userId) {
        LambdaQueryWrapper<Documents> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(Documents::getId, docId).eq(Documents::getKbId, kbId);
        Documents document = this.getOne(queryWrapper);
        if (document == null) {
            throw new BusinessException(404, "删除失败，文档不存在或没有权限");
        }
        KnowledgeBases kb = knowledgeBasesService.getById(kbId);
        if (kb == null || (!userId.equals(document.getUploaderId()) && !userId.equals(kb.getOwnerUserId()))) {
            throw new BusinessException(403, "没有权限删除该文档");
        }

        long groupId = document.getUploaderId() / 1000;
        String dbName = "group_" + groupId;
        String collectionName = "kb_" + document.getKbId();
        try {
            // 文档的删除必须保证向量库是加载状态
            GetLoadStateReq getLoadStateReq = GetLoadStateReq.builder().databaseName(dbName).collectionName(collectionName).build();
            boolean isLoaded = milvusClientV2.getLoadState(getLoadStateReq);
            if (!isLoaded) {
                // 尝试加载集合
                LoadCollectionReq loadCollectionReq = LoadCollectionReq.builder()
                        .databaseName(dbName)
                        .collectionName(collectionName)
                        .sync(true)
                        .build();
                milvusClientV2.loadCollection(loadCollectionReq);
            }
            milvusClientV2.delete(DeleteReq.builder().databaseName(dbName).collectionName(collectionName).filter("documentId == " + docId).build());
            log.info("Deleted vectors from Milvus: docId={}, db={}, collection={}", docId, dbName, collectionName);
        } catch (Exception e) {
            log.error("Failed to delete from Milvus: docId={}, error={}", docId, e.getMessage());
            throw new BusinessException(500, "向量化内容删除失败");
        }

        try {
            documentChunksService.remove(new LambdaQueryWrapper<com.rag.ragserver.domain.DocumentChunks>()
                    .eq(com.rag.ragserver.domain.DocumentChunks::getDocumentId, docId));
            this.removeById(docId);
            minioClient.removeObject(RemoveObjectArgs.builder()
                    .bucket(minioConfig.getBucketName())
                    .object(document.getFilePath())
                    .build());
        } catch (Exception e) {
            log.error("Failed to delete document data: docId={}, error={}", docId, e.getMessage());
            throw new BusinessException(500, "文档数据删除失败");
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void reprocessDocument(Long kbId, Long docId, Long userId) {
        LambdaQueryWrapper<Documents> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(Documents::getId, docId).eq(Documents::getKbId, kbId);
        Documents document = this.getOne(queryWrapper);
        if (document == null) {
            throw new BusinessException(404, "文档不存在");
        }
        if (!"failed".equals(String.valueOf(document.getStatus()))) {
            throw new BusinessException(400, "仅处理失败的文档可以重试");
        }

        boolean claimed = this.update(new UpdateWrapper<Documents>()
                .eq("id", docId)
                .eq("kb_id", kbId)
                .eq("status", "failed")
                .set("status", "processing")
                .set("error_message", ""));
        if (!claimed) {
            throw new BusinessException(400, "仅处理失败的文档可以重试");
        }

        DocumentProcessMessage message = DocumentProcessMessage.builder()
                .documentId(document.getId())
                .kbId(kbId)
                .userId(document.getUploaderId())
                .filePath(document.getFilePath())
                .fileName(document.getFileName())
                .bucketName(minioConfig.getBucketName())
                .build();
        rabbitTemplate.convertAndSend("server.interact.llm.exchange", "rag.document.process.key", message);
        log.info("Document reprocess message sent: documentId={}", document.getId());
    }

    @Override
    public void previewDocument(Long docId, Long userId, HttpServletResponse response) {
        Documents document = this.getById(docId);
        if (document == null) {
            throw new BusinessException(404, "文档不存在");
        }

        try (InputStream stream = minioClient.getObject(GetObjectArgs.builder().bucket(minioConfig.getBucketName()).object(document.getFilePath()).build())) {

            String contentType = document.getMimeType();
            if (contentType == null) {
                contentType = "application/octet-stream";
            }
            response.setContentType(contentType);

            OutputStream out = response.getOutputStream();
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = stream.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
            }
            out.flush();
        } catch (Exception e) {
            log.error("Failed to stream file content: {}", document.getFilePath(), e);
            throw new BusinessException(500, "文件预览失败");
        }
    }

    static Documents findDuplicateInFolder(List<Documents> candidates, String folderPath) {
        if (candidates == null) {
            return null;
        }
        for (Documents candidate : candidates) {
            if (folderPath.equals(folderPathOf(candidate.getFileName()))) {
                return candidate;
            }
        }
        return null;
    }

    static String folderPathOf(String fileName) {
        if (fileName == null || fileName.isEmpty()) {
            return "";
        }
        String normalized = fileName.replace('\\', '/').replaceAll("/{2,}", "/");
        int end = normalized.length();
        while (end > 0 && normalized.charAt(end - 1) == '/') {
            end--;
        }
        int start = 0;
        while (start < end && normalized.charAt(start) == '/') {
            start++;
        }
        int slash = normalized.lastIndexOf('/', end - 1);
        if (slash < start) {
            return "";
        }
        return normalized.substring(start, slash);
    }
}




