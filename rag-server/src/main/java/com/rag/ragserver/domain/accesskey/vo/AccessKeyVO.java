package com.rag.ragserver.domain.accesskey.vo;

import lombok.Data;

import java.util.Date;

@Data
public class AccessKeyVO {
    private Long id;
    private String name;
    private String keyPrefix;
    private Date createdAt;
    private Date lastUsedAt;
    private Date revokedAt;
    private boolean active;
}
