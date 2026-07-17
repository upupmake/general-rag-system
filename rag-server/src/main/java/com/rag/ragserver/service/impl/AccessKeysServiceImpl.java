package com.rag.ragserver.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.rag.ragserver.domain.AccessKeys;
import com.rag.ragserver.domain.accesskey.vo.AccessKeyCreatedVO;
import com.rag.ragserver.domain.accesskey.vo.AccessKeyVO;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.mapper.AccessKeysMapper;
import com.rag.ragserver.service.AccessKeysService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class AccessKeysServiceImpl extends ServiceImpl<AccessKeysMapper, AccessKeys>
        implements AccessKeysService {

    private static final String KEY_PREFIX = "grs_ak_";
    private static final int RANDOM_BYTES = 32;
    private static final int DISPLAY_PREFIX_LENGTH = 15;
    private final SecureRandom secureRandom = new SecureRandom();

    @Override
    public List<AccessKeyVO> listByUserId(Long userId) {
        LambdaQueryWrapper<AccessKeys> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AccessKeys::getUserId, userId)
                .orderByDesc(AccessKeys::getCreatedAt);
        return this.list(wrapper).stream()
                .map(this::toVO)
                .collect(Collectors.toList());
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public AccessKeyCreatedVO createAccessKey(Long userId, String name) {
        byte[] randomBytes = new byte[RANDOM_BYTES];
        secureRandom.nextBytes(randomBytes);
        String rawKey = KEY_PREFIX + Base64.getUrlEncoder().withoutPadding().encodeToString(randomBytes);
        Date now = new Date();

        AccessKeys accessKey = new AccessKeys();
        accessKey.setUserId(userId);
        accessKey.setName(name.trim());
        accessKey.setKeyHash(hash(rawKey));
        accessKey.setKeyPrefix(rawKey.substring(0, DISPLAY_PREFIX_LENGTH));
        accessKey.setCreatedAt(now);

        if (!this.save(accessKey)) {
            throw new BusinessException(500, "Access Key 创建失败");
        }

        AccessKeyCreatedVO result = new AccessKeyCreatedVO();
        result.setId(accessKey.getId());
        result.setName(accessKey.getName());
        result.setKeyPrefix(accessKey.getKeyPrefix());
        result.setAccessKey(rawKey);
        result.setCreatedAt(now);
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void revokeAccessKey(Long userId, Long accessKeyId) {
        LambdaUpdateWrapper<AccessKeys> wrapper = new LambdaUpdateWrapper<>();
        wrapper.eq(AccessKeys::getId, accessKeyId)
                .eq(AccessKeys::getUserId, userId)
                .isNull(AccessKeys::getRevokedAt)
                .set(AccessKeys::getRevokedAt, new Date());
        if (!this.update(wrapper)) {
            throw new BusinessException(404, "Access Key 不存在或已撤销");
        }
    }

    @Override
    public AccessKeys findActiveAccessKey(String rawKey) {
        if (rawKey == null || !rawKey.startsWith(KEY_PREFIX)) {
            return null;
        }
        LambdaQueryWrapper<AccessKeys> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AccessKeys::getKeyHash, hash(rawKey))
                .isNull(AccessKeys::getRevokedAt);
        return this.getOne(wrapper);
    }

    @Override
    public void markUsed(Long accessKeyId) {
        LambdaUpdateWrapper<AccessKeys> wrapper = new LambdaUpdateWrapper<>();
        wrapper.eq(AccessKeys::getId, accessKeyId)
                .isNull(AccessKeys::getRevokedAt)
                .set(AccessKeys::getLastUsedAt, new Date());
        this.update(wrapper);
    }

    private AccessKeyVO toVO(AccessKeys accessKey) {
        AccessKeyVO result = new AccessKeyVO();
        result.setId(accessKey.getId());
        result.setName(accessKey.getName());
        result.setKeyPrefix(accessKey.getKeyPrefix());
        result.setCreatedAt(accessKey.getCreatedAt());
        result.setLastUsedAt(accessKey.getLastUsedAt());
        result.setRevokedAt(accessKey.getRevokedAt());
        result.setActive(accessKey.getRevokedAt() == null);
        return result;
    }

    private String hash(String rawKey) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hashed = digest.digest(rawKey.getBytes(StandardCharsets.UTF_8));
            StringBuilder result = new StringBuilder(hashed.length * 2);
            for (byte value : hashed) {
                result.append(String.format("%02x", value));
            }
            return result.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }
}
