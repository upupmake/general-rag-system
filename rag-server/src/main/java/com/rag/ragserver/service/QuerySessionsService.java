package com.rag.ragserver.service;

import com.rag.ragserver.domain.QuerySessions;
import com.baomidou.mybatisplus.extension.service.IService;
import com.rag.ragserver.domain.session.vo.SessionListVO;
import com.rag.ragserver.dto.SessionCursorQuery;

import com.rag.ragserver.dto.SessionSearchResultDTO;
import java.util.List;

/**
* @author make
* @description 针对表【query_sessions(RAG 查询会话上下文表)】的数据库操作Service
* @createDate 2026-01-02 22:22:17
*/
public interface QuerySessionsService extends IService<QuerySessions> {
    String generateTitle(Long sessionId, Long userId, Long workspaceId);
    SessionListVO listByCursor(
            Long userId,
            Long workspaceId,
            SessionCursorQuery query
    );
    Boolean deleteSession(Long sessionId, Long userId, Long workspaceId);
    String renameSession(Long sessionId, Long userId, Long workspaceId, String title);

    List<SessionSearchResultDTO> searchSessions(Long userId, Long workspaceId, String keyword, int limit, int offset);
}
