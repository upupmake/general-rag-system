package com.rag.ragserver.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.rag.ragserver.common.R;
import com.rag.ragserver.domain.QuerySessions;
import com.rag.ragserver.domain.session.vo.SessionListVO;
import com.rag.ragserver.dto.SessionCursorQuery;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.service.QuerySessionsService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import com.rag.ragserver.dto.SessionSearchResultDTO;
import java.util.List;
import java.util.Map;

import com.rag.ragserver.dto.SessionSearchQuery;
import org.springframework.validation.annotation.Validated;

@RestController
@RequestMapping("/sessions")
@RequiredArgsConstructor
public class QuerySessionController {
    private final QuerySessionsService querySessionsService;
    private final HttpServletRequest request;

    @PostMapping("/search")
    public R<List<SessionSearchResultDTO>> searchSessions(
        @RequestBody @Validated SessionSearchQuery query
    ) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        
        return R.success(
            querySessionsService.searchSessions(userId, workspaceId, query.getKeyword(), query.getLimit(), query.getOffset())
        );
    }

    @PostMapping("/list")
    public R<SessionListVO> listSessions(
            @RequestBody SessionCursorQuery query
    ) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");

        return R.success(
                querySessionsService.listByCursor(userId, workspaceId, query)
        );
    }

    @DeleteMapping("/{sessionId}")
    public R<Void> deleteSession(@PathVariable Long sessionId) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");

        Boolean f = querySessionsService.deleteSession(sessionId, userId, workspaceId);
        if (f) {
            return R.success();
        }
        throw new BusinessException(404, "会话不存在或无权限删除");
    }

    @GetMapping("/{sessionId}/title")
    public R<Map<String, String>> getSessionTitle(@PathVariable Long sessionId) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        String title = querySessionsService.generateTitle(sessionId, userId, workspaceId);
        return R.success(Map.of("title", title));
    }

    @PutMapping("/{sessionId}/title")
    public R<Map<String, String>> renameSession(
            @PathVariable Long sessionId,
            @RequestBody Map<String, String> body
    ) {
        Long userId = (Long) request.getAttribute("userId");
        Long workspaceId = (Long) request.getAttribute("workspaceId");
        String newTitle = querySessionsService.renameSession(sessionId, userId, workspaceId, body.get("title"));
        return R.success(Map.of("title", newTitle));
    }
}
