package com.rag.ragserver.mapper;

import com.rag.ragserver.domain.ConversationMessages;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import com.rag.ragserver.domain.model.vo.ModelPerformanceStats;
import java.util.Date;
import java.util.List;

/**
* @author make
* @description 针对表【conversation_messages(RAG 对话消息历史表)】的数据库操作Mapper
* @createDate 2026-01-02 23:06:15
* @Entity com.rag.ragserver.domain.ConversationMessages
*/
public interface ConversationMessagesMapper extends BaseMapper<ConversationMessages> {

    @Select("SELECT SUM(total_tokens) FROM conversation_messages WHERE created_at >= #{startTime}")
    Long sumTotalTokensSince(@Param("startTime") Date startTime);

    @Select("SELECT SUM(total_tokens) FROM conversation_messages WHERE created_at >= #{startTime} AND user_id = #{userId}")
    Long sumUserTotalTokensSince(@Param("startTime") Date startTime, @Param("userId") Long userId);

    @Select("SELECT m.id as modelId, m.name as modelName, m.provider, " +
            "COALESCE(s.requestCount, 0) as requestCount, " +
            "COALESCE(s.avgLatency, 100) as avgLatency, " +
            "COALESCE(s.successRate, 100.0) as successRate " +
            "FROM models m " +
            "LEFT JOIN (" +
            "  SELECT model_id, COUNT(*) as requestCount, AVG(first_token_latency_ms) as avgLatency, " +
            "  SUM(CASE WHEN is_success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as successRate " +
            "  FROM conversation_messages " +
            "  WHERE role = 'assistant' AND created_at >= #{startTime} AND model_id IS NOT NULL " +
            "  GROUP BY model_id" +
            ") s ON m.id = s.model_id " +
            "WHERE m.enabled = 1")
    List<ModelPerformanceStats> getModelPerformanceStats(@Param("startTime") Date startTime);
}




