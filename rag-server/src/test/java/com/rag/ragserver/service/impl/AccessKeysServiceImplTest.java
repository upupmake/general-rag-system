package com.rag.ragserver.service.impl;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.rag.ragserver.domain.AccessKeys;
import com.rag.ragserver.domain.accesskey.vo.AccessKeyCreatedVO;
import com.rag.ragserver.mapper.AccessKeysMapper;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AccessKeysServiceImplTest {
    private AccessKeysMapper mapper;
    private AccessKeysServiceImpl service;

    @BeforeEach
    void setUp() {
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(new MybatisConfiguration(), ""), AccessKeys.class);
        mapper = mock(AccessKeysMapper.class);
        service = new AccessKeysServiceImpl();
        ReflectionTestUtils.setField(service, "baseMapper", mapper);
        when(mapper.insert(any(AccessKeys.class))).thenAnswer(invocation -> {
            AccessKeys accessKey = invocation.getArgument(0);
            accessKey.setId(1L);
            return 1;
        });
    }

    @Test
    void createAccessKeyStoresHashInsteadOfPlaintext() throws Exception {
        AccessKeyCreatedVO created = service.createAccessKey(42L, " MCP 服务 ");

        ArgumentCaptor<AccessKeys> captor = ArgumentCaptor.forClass(AccessKeys.class);
        verify(mapper).insert(captor.capture());
        AccessKeys stored = captor.getValue();

        assertTrue(created.getAccessKey().startsWith("grs_ak_"));
        assertEquals("MCP 服务", stored.getName());
        assertEquals(42L, stored.getUserId());
        assertEquals(64, stored.getKeyHash().length());
        assertFalse(stored.getKeyHash().contains(created.getAccessKey()));
        assertEquals(sha256(created.getAccessKey()), stored.getKeyHash());
        assertEquals(created.getAccessKey().substring(0, 15), stored.getKeyPrefix());
    }

    @Test
    @SuppressWarnings("unchecked")
    void findActiveAccessKeyRejectsWrongPrefixAndQueriesHash() {
        assertNull(service.findActiveAccessKey("wrong_key"));

        AccessKeys stored = new AccessKeys();
        stored.setId(2L);
        when(mapper.selectOne(any(Wrapper.class), anyBoolean())).thenReturn(stored);

        AccessKeys result = service.findActiveAccessKey("grs_ak_test");

        assertSame(stored, result);
        verify(mapper).selectOne(any(Wrapper.class), anyBoolean());
    }

    @Test
    @SuppressWarnings("unchecked")
    void listByUserIdExcludesRevokedAccessKeys() {
        when(mapper.selectList(any(Wrapper.class))).thenReturn(java.util.Collections.emptyList());

        service.listByUserId(42L);

        ArgumentCaptor<Wrapper<AccessKeys>> captor = ArgumentCaptor.forClass(Wrapper.class);
        verify(mapper).selectList(captor.capture());
        assertTrue(captor.getValue().getSqlSegment().toUpperCase().contains("REVOKED_AT IS NULL"));
    }

    private String sha256(String value) throws Exception {
        byte[] hashed = MessageDigest.getInstance("SHA-256")
                .digest(value.getBytes(StandardCharsets.UTF_8));
        StringBuilder result = new StringBuilder(hashed.length * 2);
        for (byte item : hashed) {
            result.append(String.format("%02x", item));
        }
        return result.toString();
    }
}
