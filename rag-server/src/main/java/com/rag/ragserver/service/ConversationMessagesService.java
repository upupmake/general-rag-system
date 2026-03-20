package com.rag.ragserver.service;

import com.rag.ragserver.domain.ConversationMessages;
import com.baomidou.mybatisplus.extension.service.IService;

import java.util.List;

/**
* @author make
* @description 针对表【conversation_messages(RAG 对话消息历史表)】的数据库操作Service
* @createDate 2026-01-02 23:06:15
*/
public interface ConversationMessagesService extends IService<ConversationMessages> {

    /**
     * 编辑最后一轮用户问题
     * @param sessionId 会话ID
     * @param messageId 用户消息ID
     * @param userId 用户ID
     * @param newContent 新的问题内容
     * @return 更新后的用户消息
     */
    ConversationMessages editLastUserMessage(Long sessionId, Long messageId, Long userId, String newContent);

    /**
     * 重试最后一轮AI回复
     * @param sessionId 会话ID
     * @param userMessageId 最后一轮用户消息ID（用于定位需要重新生成回复的用户问题）
     * @param userId 用户ID
     * @return 用户消息（用于重新生成回复）
     */
    ConversationMessages retryLastAssistantMessage(Long sessionId, Long userMessageId, Long userId);

    Long countTodayTokens();

    Long countTodayTokens(Long userId);
    List<ConversationMessages> getLastNRagContextMessages(Long sessionId, int n);
}
