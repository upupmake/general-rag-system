package com.rag.ragserver.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.rag.ragserver.domain.ConversationMessages;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.service.ConversationMessagesService;
import com.rag.ragserver.mapper.ConversationMessagesMapper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.ZoneId;
import java.util.Date;
import java.util.List;

/**
 * @author make
 * @description 针对表【conversation_messages(RAG 对话消息历史表)】的数据库操作Service实现
 * @createDate 2026-01-02 23:06:15
 */
@Service
public class ConversationMessagesServiceImpl extends ServiceImpl<ConversationMessagesMapper, ConversationMessages>
        implements ConversationMessagesService {

    @Override
    public Long countTodayTokens() {
        LocalDate today = LocalDate.now();
        Date startOfDay = Date.from(today.atStartOfDay(ZoneId.systemDefault()).toInstant());
        Long total = this.baseMapper.sumTotalTokensSince(startOfDay);
        return total != null ? total : 0L;
    }

    @Override
    public Long countTodayTokens(Long userId) {
        LocalDate today = LocalDate.now();
        Date startOfDay = Date.from(today.atStartOfDay(ZoneId.systemDefault()).toInstant());
        Long total = this.baseMapper.sumUserTotalTokensSince(startOfDay, userId);
        return total != null ? total : 0L;
    }

    @Override
    public List<ConversationMessages> getLastNRagContextMessages(Long sessionId, int n) {
        LambdaQueryWrapper<ConversationMessages> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.select(ConversationMessages::getId, ConversationMessages::getRagContext)
                .eq(ConversationMessages::getSessionId, sessionId)
                .eq(ConversationMessages::getRole, "assistant")
                .orderByDesc(ConversationMessages::getCreatedAt);
        IPage<ConversationMessages> pageWrapper = new Page<>(1, n);
        IPage<ConversationMessages> page = this.page(pageWrapper, queryWrapper);
        return page.getRecords();
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ConversationMessages editLastUserMessage(Long sessionId, Long messageId, Long userId, String newContent) {
        // 1. 获取该会话的所有未删除消息，按创建时间升序排列
        LambdaQueryWrapper<ConversationMessages> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(ConversationMessages::getSessionId, sessionId)
                .eq(ConversationMessages::getUserId, userId)
                .and(w -> w.isNull(ConversationMessages::getIsDeleted).or().eq(ConversationMessages::getIsDeleted, 0))
                .orderByAsc(ConversationMessages::getCreatedAt);
        List<ConversationMessages> messages = this.list(queryWrapper);

        if (messages.isEmpty()) {
            throw new BusinessException(404, "会话不存在或无消息");
        }

        // 2. 找到最后一条用户消息
        ConversationMessages lastUserMessage = null;
        int lastUserIndex = -1;
        for (int i = messages.size() - 1; i >= 0; i--) {
            if ("user".equals(messages.get(i).getRole())) {
                lastUserMessage = messages.get(i);
                lastUserIndex = i;
                break;
            }
        }

        if (lastUserMessage == null) {
            throw new BusinessException(400, "没有找到用户消息");
        }

        // 3. 校验：只允许编辑最后一轮的用户消息
        if (!lastUserMessage.getId().equals(messageId)) {
            throw new BusinessException(400, "只能编辑最后一轮对话的用户问题");
        }

        // 4. 校验状态：非generating状态才能编辑
        String status = lastUserMessage.getStatus() != null ? lastUserMessage.getStatus().toString() : "pending";
        if ("generating".equals(status)) {
            throw new BusinessException(400, "AI正在生成回复中，请稍后再试");
        }

        // 5. 逻辑删除该用户消息对应的assistant消息（如果存在）
        if (lastUserIndex < messages.size() - 1) {
            ConversationMessages nextMessage = messages.get(lastUserIndex + 1);
            if ("assistant".equals(nextMessage.getRole())) {
                this.removeById(nextMessage.getId());
            }
        }

        // 6. 更新用户消息内容和状态
        lastUserMessage.setContent(newContent);
        lastUserMessage.setStatus("pending");
        this.updateById(lastUserMessage);

        return lastUserMessage;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ConversationMessages retryLastAssistantMessage(Long sessionId, Long userMessageId, Long userId) {
        // 1. 获取该会话的所有未删除消息，按创建时间升序排列
        LambdaQueryWrapper<ConversationMessages> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(ConversationMessages::getSessionId, sessionId)
                .eq(ConversationMessages::getUserId, userId)
                .orderByAsc(ConversationMessages::getCreatedAt);
        List<ConversationMessages> messages = this.list(queryWrapper);

        if (messages.isEmpty()) {
            throw new BusinessException(404, "会话不存在或无消息");
        }

        // 2. 找到最后一条用户消息
        ConversationMessages lastUserMessage = null;
        int lastUserIndex = -1;
        for (int i = messages.size() - 1; i >= 0; i--) {
            if ("user".equals(messages.get(i).getRole())) {
                lastUserMessage = messages.get(i);
                lastUserIndex = i;
                break;
            }
        }

        if (lastUserMessage == null) {
            throw new BusinessException(400, "没有找到用户消息");
        }

        // 3. 校验：只允许重试最后一轮的用户消息对应的回复
        if (!lastUserMessage.getId().equals(userMessageId)) {
            throw new BusinessException(400, "只能重试最后一轮对话的AI回复");
        }

        // 4. 校验状态：非generating状态才能重试
        String status = lastUserMessage.getStatus() != null ? lastUserMessage.getStatus().toString() : "pending";
        if ("generating".equals(status)) {
            throw new BusinessException(400, "AI正在生成回复中，请稍后再试");
        }

        // 5. 校验：必须存在对应的assistant消息才能重试
        if (lastUserIndex >= messages.size() - 1) {
            throw new BusinessException(400, "没有可重试的AI回复");
        }

        ConversationMessages assistantMessage = messages.get(lastUserIndex + 1);
        if (!"assistant".equals(assistantMessage.getRole())) {
            throw new BusinessException(400, "没有可重试的AI回复");
        }

        // 6. 逻辑删除该assistant消息
        assistantMessage.setIsDeleted(1);
        this.removeById(assistantMessage.getId());

        // 7. 将用户消息状态改为pending，等待重新生成回复
        lastUserMessage.setStatus("pending");
        this.updateById(lastUserMessage);

        return lastUserMessage;
    }
}




