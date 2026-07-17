package com.rag.ragserver.domain.accesskey.vo;

import lombok.Data;

import java.util.Date;

@Data
public class AccessKeyCreatedVO {
    private Long id;
    private String name;
    private String keyPrefix;
    private String accessKey;
    private Date createdAt;
}
