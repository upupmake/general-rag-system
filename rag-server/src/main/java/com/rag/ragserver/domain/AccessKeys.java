package com.rag.ragserver.domain;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.util.Date;

@Data
@TableName("access_keys")
public class AccessKeys {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long userId;
    private String name;
    private String keyHash;
    private String keyPrefix;
    private Date createdAt;
    private Date lastUsedAt;
    private Date revokedAt;
}
