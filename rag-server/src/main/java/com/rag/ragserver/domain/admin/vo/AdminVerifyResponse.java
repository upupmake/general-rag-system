package com.rag.ragserver.domain.admin.vo;

import lombok.Data;

@Data
public class AdminVerifyResponse {

    private String token;

    /**
     * 有效期，单位秒。
     */
    private Long expiresIn;

    /**
     * 过期时间，毫秒时间戳。
     */
    private Long expiresAt;
}
