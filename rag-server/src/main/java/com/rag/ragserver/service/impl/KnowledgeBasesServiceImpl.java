package com.rag.ragserver.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.rag.ragserver.domain.KnowledgeBases;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.service.KbSharesService;
import com.rag.ragserver.service.KnowledgeBasesService;
import com.rag.ragserver.mapper.KnowledgeBasesMapper;
import com.rag.ragserver.mapper.KbSharesMapper;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.AddFieldReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import io.milvus.v2.service.collection.request.DropCollectionReq;
import io.milvus.v2.service.collection.request.HasCollectionReq;
import io.milvus.v2.service.database.request.CreateDatabaseReq;
import io.milvus.v2.service.database.response.ListDatabasesResp;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author make
 * @description 针对表【knowledge_bases(知识库主表)】的数据库操作Service实现
 * @createDate 2025-12-31 02:01:56
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class KnowledgeBasesServiceImpl extends ServiceImpl<KnowledgeBasesMapper, KnowledgeBases>
        implements KnowledgeBasesService {

    private final MilvusClientV2 milvusClientV2;
    private final KbSharesMapper kbSharesMapper;

    @Override
    public Map<String, List<KnowledgeBases>> listByWorkspaceAndUser(Long workspaceId, Long userId) {
        LambdaQueryWrapper<KnowledgeBases> queryWrapper = new LambdaQueryWrapper<>();
        // 分为4类：1是自己创建的，2共同workspace下的，3public的，4被邀请访问的
        queryWrapper
                .or(w -> w.eq(KnowledgeBases::getOwnerUserId, userId).eq(KnowledgeBases::getVisibility, "private"))
                .or(w -> w.eq(KnowledgeBases::getWorkspaceId, workspaceId).eq(KnowledgeBases::getVisibility, "shared"))
                .or(w -> w.eq(KnowledgeBases::getVisibility, "public"))
                .orderByDesc(KnowledgeBases::getCreatedAt);
        List<KnowledgeBases> kbList = this.list(queryWrapper);

        // 查询被邀请访问的知识库 - 直接使用 Mapper 避免循环依赖
        LambdaQueryWrapper<com.rag.ragserver.domain.KbShares> sharesWrapper = new LambdaQueryWrapper<>();
        sharesWrapper.eq(com.rag.ragserver.domain.KbShares::getUserId, userId);
        List<com.rag.ragserver.domain.KbShares> shares = kbSharesMapper.selectList(sharesWrapper);
        List<Long> invitedKbIds = shares.stream()
                .map(com.rag.ragserver.domain.KbShares::getKbId)
                .collect(java.util.stream.Collectors.toList());

        List<KnowledgeBases> invitedKbs = new ArrayList<>();
        if (!invitedKbIds.isEmpty()) {
            LambdaQueryWrapper<KnowledgeBases> invitedWrapper = new LambdaQueryWrapper<>();
            invitedWrapper.in(KnowledgeBases::getId, invitedKbIds)
                    .orderByDesc(KnowledgeBases::getCreatedAt);
            invitedKbs = this.list(invitedWrapper);
        }

        // 将数据分为四类
        Map<String, List<KnowledgeBases>> result = new java.util.HashMap<>();
        result.put("private", new ArrayList<>());
        result.put("shared", new ArrayList<>());
        result.put("public", new ArrayList<>());
        result.put("invited", new ArrayList<>());

        kbList.forEach(kb -> {
            if (workspaceId != null && workspaceId.equals(kb.getWorkspaceId())) {
                result.get("shared").add(kb);
            } else if ("public".equals(kb.getVisibility())) {
                result.get("public").add(kb);
            } else if (userId.equals(kb.getOwnerUserId())) {
                result.get("private").add(kb);
            }
        });

        result.put("invited", invitedKbs);

        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public KnowledgeBases createKnowledgeBase(KnowledgeBases kb) {
        // 1. 先保存到 MySQL，获取自增 ID
        boolean saved = this.save(kb);
        if (!saved) {
            throw new BusinessException(500, "知识库创建失败");
        }

        Long kbId = kb.getId();
        Long userId = kb.getOwnerUserId();
        long groupId = userId / 1000;
        String dbName = "group_" + groupId;
        String collectionName = "kb_" + kbId;

        try {
            // 查询是否存在当前Milvus数据库
            ListDatabasesResp listDatabasesResp = milvusClientV2.listDatabases();
            boolean dbExists = listDatabasesResp.getDatabaseNames().contains(dbName);
            if (!dbExists) {
                // 不存在就创建一个数据库
                CreateDatabaseReq createDatabaseReq = CreateDatabaseReq.builder()
                        .databaseName(dbName)
                        .build();
                milvusClientV2.createDatabase(createDatabaseReq);
            }
            // 创建一个 Milvus 集合用于存储向量 注意创建索引
            CreateCollectionReq.CollectionSchema schema = CreateCollectionReq.CollectionSchema.builder()
                    .build();
            // 创建四个字段，text,vector,pk,documentId
            schema.addField(
                    AddFieldReq.builder()
                            .fieldName("text")
                            .dataType(DataType.VarChar)
                            .maxLength(6144)
                            .build()
            );
            schema.addField(
                    AddFieldReq.builder()
                            .fieldName("vector")
                            .dataType(DataType.FloatVector)
                            .dimension(1024) // 假设使用1024维的向量
                            .build()
            );
            schema.addField(
                    AddFieldReq.builder()
                            .fieldName("pk")
                            .dataType(DataType.Int64)
                            .isPrimaryKey(true)
                            .autoID(true)
                            .build()
            );
            schema.addField(
                    AddFieldReq.builder()
                            .fieldName("documentId")
                            .dataType(DataType.Int64)
                            .build()
            );
            schema.addField(
                    AddFieldReq.builder()
                            .fieldName("chunkIndex")
                            .dataType(DataType.Int64)
                            .build()
            );
            schema.addField(
                    AddFieldReq.builder()
                            .fieldName("maxChunkIndex")
                            .dataType(DataType.Int64)
                            .build()
            );
            schema.addField(
                    AddFieldReq.builder()
                            .fieldName("fileName")
                            .dataType(DataType.VarChar)
                            .maxLength(1024)
                            .build()
            );
            // 创建索引
            List<IndexParam> indexParams = new ArrayList<>();
            indexParams.add(
                    IndexParam.builder()
                            .fieldName("vector")
                            .indexType(IndexParam.IndexType.AUTOINDEX)
                            .metricType(IndexParam.MetricType.COSINE)
                            .build()
            );
            indexParams.add(
                    IndexParam.builder()
                            .fieldName("pk")
                            .indexType(IndexParam.IndexType.AUTOINDEX)
                            .build()
            );
            indexParams.add(
                    IndexParam.builder()
                            .fieldName("documentId")
                            .indexType(IndexParam.IndexType.STL_SORT)
                            .build()
            );
            indexParams.add(
                    IndexParam.builder()
                            .fieldName("text")
                            .indexType(IndexParam.IndexType.NGRAM)
                            .extraParams(Map.of(
                                    "min_gram", 2,
                                    "max_gram", 3
                            ))
                            .build()
            );
            indexParams.add(
                    IndexParam.builder()
                            .fieldName("fileName")
                            .indexType(IndexParam.IndexType.INVERTED)
                            .build()
            );
            milvusClientV2.createCollection(
                    CreateCollectionReq.builder()
                            .databaseName(dbName)
                            .collectionName(collectionName)
                            .collectionSchema(schema)
                            .indexParams(indexParams)
                            .build()
            );
        } catch (Exception e) {
            log.error("Failed to create Milvus collection: db={}, collection={}, error={}",
                    dbName, collectionName, e.getMessage());
            throw new BusinessException(500, "向量数据库初始化失败");
        }

        return kb;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteKnowledgeBase(Long kbId, Long userId) {
        // 1. 验证知识库是否存在且属于该用户
        LambdaQueryWrapper<KnowledgeBases> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(KnowledgeBases::getId, kbId)
                .eq(KnowledgeBases::getOwnerUserId, userId);
        KnowledgeBases kb = this.getOne(queryWrapper);

        if (kb == null) {
            throw new BusinessException(404, "知识库不存在或无权删除");
        }

        // 2. 删除 Milvus 中的集合
        long groupId = userId / 1000;
        String dbName = "group_" + groupId;
        String collectionName = "kb_" + kbId;

        try {
            // 检查集合是否存在
            boolean hasCollection = milvusClientV2.hasCollection(
                    HasCollectionReq.builder()
                            .databaseName(dbName)
                            .collectionName(collectionName)
                            .build()
            );

            if (hasCollection) {
                milvusClientV2.dropCollection(
                        DropCollectionReq.builder()
                                .databaseName(dbName)
                                .collectionName(collectionName)
                                .build()
                );
                log.info("Dropped Milvus collection: db={}, collection={}", dbName, collectionName);
                // 3. 删除数据库记录
                boolean removed = this.removeById(kbId);
                if (!removed) {
                    throw new BusinessException(500, "知识库删除失败");
                }
            }
        } catch (Exception e) {
            log.warn("Failed to drop Milvus collection: db={}, collection={}, error={}",
                    dbName, collectionName, e.getMessage());
            // 继续执行数据库删除，避免造成数据残留
        }
    }
}




