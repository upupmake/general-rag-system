package com.rag.ragserver.controller;

import com.rag.ragserver.common.R;
import com.rag.ragserver.domain.admin.vo.AdminMemberVO;
import com.rag.ragserver.domain.admin.vo.AdminVerifyResponse;
import com.rag.ragserver.domain.admin.vo.MembershipGrantResultVO;
import com.rag.ragserver.dto.AdminPasswordRequest;
import com.rag.ragserver.dto.MembershipBzUpdateRequest;
import com.rag.ragserver.dto.MembershipGrantRequest;
import com.rag.ragserver.service.AdminMembershipService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.validation.Valid;
import java.util.List;

@RestController
@RequestMapping("/admin/membership")
@RequiredArgsConstructor
public class AdminMembershipController {

    private static final String ADMIN_TOKEN_HEADER = "X-Admin-Token";

    private final AdminMembershipService adminMembershipService;

    @PostMapping("/verify")
    public R<AdminVerifyResponse> verify(@RequestBody @Valid AdminPasswordRequest request) {
        return R.success(adminMembershipService.verifyPassword(request.getPassword()));
    }

    @GetMapping("/members")
    public R<List<AdminMemberVO>> members(
            @RequestHeader(value = ADMIN_TOKEN_HEADER, required = false) String adminToken) {
        return R.success(adminMembershipService.listMembers(adminToken));
    }

    @PostMapping("/grant")
    public R<MembershipGrantResultVO> grant(
            @RequestHeader(value = ADMIN_TOKEN_HEADER, required = false) String adminToken,
            @RequestBody @Valid MembershipGrantRequest request) {
        return R.success(adminMembershipService.grant(adminToken, request));
    }

    @PutMapping("/{userId}/bz")
    public R<AdminMemberVO> updateBz(
            @RequestHeader(value = ADMIN_TOKEN_HEADER, required = false) String adminToken,
            @PathVariable Long userId,
            @RequestBody @Valid MembershipBzUpdateRequest request) {
        return R.success(adminMembershipService.updateBz(adminToken, userId, request));
    }

    @PostMapping("/logout")
    public R<String> logout(
            @RequestHeader(value = ADMIN_TOKEN_HEADER, required = false) String adminToken) {
        adminMembershipService.logout(adminToken);
        return R.success("已退出管理");
    }
}
