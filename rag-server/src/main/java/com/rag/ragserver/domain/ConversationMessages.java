package com.rag.ragserver.domain;

import com.baomidou.mybatisplus.annotation.*;

import java.util.Date;

import lombok.Data;

/**
 * RAG 对话消息历史表
 *
 * @TableName conversation_messages
 */
@TableName(value = "conversation_messages", autoResultMap = true)
@Data
public class ConversationMessages {
    /**
     * 对话消息 ID
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 所属会话 ID，对应 query_sessions.id
     */
    private Long sessionId;

    /**
     * 用户 ID（冗余字段，便于查询与审计）
     */
    private Long userId;

    /**
     * 当前使用的知识库 ID（冗余字段）
     */
    private Long kbId;

    /**
     * 消息角色：user / assistant / system
     */
    private Object role;

    /**
     * 消息文本内容
     */
    private String content;

    /**
     * 消息的状态 'pending','generating','completed','aborted','error'
     */
    private Object status;

    /**
     * 本次生成使用的模型（assistant 消息才有）
     */
    private Long modelId;

    /**
     * prompt token 数
     */
    private Integer promptTokens;

    /**
     * completion token 数
     */
    private Integer completionTokens;

    /**
     * 总 token 数
     */
    private Integer totalTokens;

    /**
     * RAG 检索上下文信息（命中的 chunk / doc / score 等）
     */
    private Object ragContext;

    /**
     * 本次生成耗时（毫秒）
     */
    private Long latencyMs;

    /**
     * 消息创建时间
     */
    private Date createdAt;

    /**
     *
     */
    @TableLogic
    private Integer isDeleted;

    /**
     *
     */
    private String thinking;

    @TableField(typeHandler = com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler.class)
    private Object options;


    private Long firstTokenLatencyMs;

    private Boolean isSuccess;
}