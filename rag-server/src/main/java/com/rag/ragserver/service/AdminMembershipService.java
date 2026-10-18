package com.rag.ragserver.service;

import com.rag.ragserver.domain.admin.vo.AdminMemberVO;
import com.rag.ragserver.domain.admin.vo.AdminVerifyResponse;
import com.rag.ragserver.domain.admin.vo.MembershipGrantResultVO;
import com.rag.ragserver.dto.MembershipBzUpdateRequest;
import com.rag.ragserver.dto.MembershipGrantRequest;

import java.util.List;

public interface AdminMembershipService {

    AdminVerifyResponse verifyPassword(String password);

    List<AdminMemberVO> listMembers(String adminToken);

    MembershipGrantResultVO grant(String adminToken, MembershipGrantRequest request);

    AdminMemberVO updateBz(String adminToken, Long userId, MembershipBzUpdateRequest request);

    void logout(String adminToken);
}
