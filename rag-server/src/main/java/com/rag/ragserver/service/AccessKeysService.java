package com.rag.ragserver.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.rag.ragserver.domain.AccessKeys;
import com.rag.ragserver.domain.accesskey.vo.AccessKeyCreatedVO;
import com.rag.ragserver.domain.accesskey.vo.AccessKeyVO;

import java.util.List;

public interface AccessKeysService extends IService<AccessKeys> {
    List<AccessKeyVO> listByUserId(Long userId);

    AccessKeyCreatedVO createAccessKey(Long userId, String name);

    void revokeAccessKey(Long userId, Long accessKeyId);

    AccessKeys findActiveAccessKey(String rawKey);

    void markUsed(Long accessKeyId);
}
