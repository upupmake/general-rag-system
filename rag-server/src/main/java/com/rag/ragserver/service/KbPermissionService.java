package com.rag.ragserver.service;

import java.util.Map;

/**
 * 知识库权限检查服务
 */
public interface KbPermissionService {
    
    /**
     * 检查用户是否可以读取知识库
     * @param kbId 知识库ID
     * @param userId 用户ID
     * @param workspaceId 用户当前工作空间ID
     * @return true-有权限, false-无权限
     */
    boolean canReadKb(Long kbId, Long userId, Long workspaceId);

    /**
     * 按所有工作空间成员关系检查读取来源。
     */
    String getReadAccessSource(Long kbId, Long userId);

    /**
     * 获取用户全部可读知识库及其唯一归类。
     */
    Map<Long, String> listReadableKbAccessSources(Long userId);
    
    /**
     * 检查用户是否可以写入知识库（上传文档等）
     * @param kbId 知识库ID
     * @param userId 用户ID
     * @param workspaceId 用户当前工作空间ID
     * @return true-有权限, false-无权限
     */
    boolean canWriteKb(Long kbId, Long userId, Long workspaceId);
    
    /**
     * 检查用户是否为知识库拥有者
     * @param kbId 知识库ID
     * @param userId 用户ID
     * @return true-是拥有者, false-不是拥有者
     */
    boolean isKbOwner(Long kbId, Long userId);
    
    /**
     * 检查用户是否为文档上传者或知识库拥有者
     * @param docId 文档ID
     * @param userId 用户ID
     * @return true-有权限, false-无权限
     */
    boolean canModifyDocument(Long docId, Long userId);
}
