package com.rag.ragserver.utils;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 会话级“流式生成中”标记。
 * 以 Redis 为准（多实例可见），TTL 兜底：进程崩溃后标记自然过期，
 * 用于区分“真正在生成”与“崩溃后遗留的 generating 状态”。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class ChatStreamRegistry {

    /** 标记有效期：期间无任何事件续期则视为流已结束（崩溃兜底） */
    private static final Duration ACTIVE_TTL = Duration.ofMinutes(5);
    /** 续期间隔，避免每个 SSE 事件都写 Redis */
    private static final long REFRESH_INTERVAL_MS = 30_000L;

    private final StringRedisTemplate redisTemplate;
    private final Map<Long, String> activeValues = new ConcurrentHashMap<>();
    private final Map<Long, Long> lastRefreshAt = new ConcurrentHashMap<>();

    private String key(Long sessionId) {
        return "chat:streaming:session:" + sessionId;
    }

    /** 流开始：写入活跃标记 */
    public void markActive(Long sessionId, Long userMessageId) {
        activeValues.put(sessionId, String.valueOf(userMessageId));
        lastRefreshAt.put(sessionId, System.currentTimeMillis());
        setKey(sessionId);
    }

    /** 流进行中：按会话节流续期（重新 set，保证 TTL 过期后也能恢复） */
    public void refresh(Long sessionId) {
        long now = System.currentTimeMillis();
        Long last = lastRefreshAt.get(sessionId);
        if (last != null && now - last < REFRESH_INTERVAL_MS) {
            return;
        }
        lastRefreshAt.put(sessionId, now);
        setKey(sessionId);
    }

    /** 流结束（完成/取消/异常）：清除标记 */
    public void clear(Long sessionId) {
        activeValues.remove(sessionId);
        lastRefreshAt.remove(sessionId);
        try {
            redisTemplate.delete(key(sessionId));
        } catch (Exception e) {
            log.warn("清除流式标记失败, sessionId={}: {}", sessionId, e.getMessage());
        }
    }

    /** Redis 异常时按“仍在生成”处理，避免误判为陈旧而重复生成 */
    public boolean isActive(Long sessionId) {
        try {
            return Boolean.TRUE.equals(redisTemplate.hasKey(key(sessionId)));
        } catch (Exception e) {
            log.warn("读取流式标记失败, sessionId={}: {}", sessionId, e.getMessage());
            return true;
        }
    }

    private void setKey(Long sessionId) {
        try {
            redisTemplate.opsForValue().set(key(sessionId), activeValues.getOrDefault(sessionId, "1"), ACTIVE_TTL);
        } catch (Exception e) {
            log.warn("写入流式标记失败, sessionId={}: {}", sessionId, e.getMessage());
        }
    }
}
