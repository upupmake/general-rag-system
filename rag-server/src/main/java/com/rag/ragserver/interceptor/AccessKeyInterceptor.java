package com.rag.ragserver.interceptor;

import com.rag.ragserver.domain.AccessKeys;
import com.rag.ragserver.domain.Users;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.service.AccessKeysService;
import com.rag.ragserver.service.UsersService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@Component
@RequiredArgsConstructor
public class AccessKeyInterceptor implements HandlerInterceptor {
    private static final String BEARER_PREFIX = "Bearer ";

    private final AccessKeysService accessKeysService;
    private final UsersService usersService;

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
        if ("OPTIONS".equals(request.getMethod())) {
            return true;
        }

        String authorization = request.getHeader("Authorization");
        if (authorization == null || !authorization.startsWith(BEARER_PREFIX)) {
            throw new BusinessException(401, "Access Key 无效");
        }

        AccessKeys accessKey = accessKeysService.findActiveAccessKey(authorization.substring(BEARER_PREFIX.length()));
        if (accessKey == null) {
            throw new BusinessException(401, "Access Key 无效或已撤销");
        }

        Users user = usersService.getById(accessKey.getUserId());
        if (user == null || !"active".equals(user.getStatus())) {
            throw new BusinessException(403, "用户不存在或已被禁用");
        }

        request.setAttribute("userId", user.getId());
        request.setAttribute("accessKeyId", accessKey.getId());
        accessKeysService.markUsed(accessKey.getId());
        return true;
    }
}
