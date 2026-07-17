package com.rag.ragserver.service.impl;

import com.rag.ragserver.domain.AccessKeys;
import com.rag.ragserver.domain.accesskey.vo.AccessKeyCreatedVO;
import com.rag.ragserver.mapper.AccessKeysMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AccessKeysServiceImplTest {
    private AccessKeysMapper mapper;
    private AccessKeysServiceImpl service;

    @BeforeEach
    void setUp() {
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
