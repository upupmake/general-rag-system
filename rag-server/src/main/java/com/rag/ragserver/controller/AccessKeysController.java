package com.rag.ragserver.controller;

import com.rag.ragserver.common.R;
import com.rag.ragserver.domain.accesskey.vo.AccessKeyCreatedVO;
import com.rag.ragserver.domain.accesskey.vo.AccessKeyVO;
import com.rag.ragserver.dto.AccessKeyCreateDTO;
import com.rag.ragserver.service.AccessKeysService;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.servlet.http.HttpServletRequest;
import java.util.List;

@RestController
@RequestMapping("/access-keys")
@RequiredArgsConstructor
public class AccessKeysController {
    private final HttpServletRequest request;
    private final AccessKeysService accessKeysService;

    @GetMapping
    public R<List<AccessKeyVO>> listAccessKeys() {
        Long userId = (Long) request.getAttribute("userId");
        return R.success(accessKeysService.listByUserId(userId));
    }

    @PostMapping
    public R<AccessKeyCreatedVO> createAccessKey(@RequestBody @Validated AccessKeyCreateDTO dto) {
        Long userId = (Long) request.getAttribute("userId");
        return R.success(accessKeysService.createAccessKey(userId, dto.getName()));
    }

    @DeleteMapping("/{accessKeyId}")
    public R<Void> revokeAccessKey(@PathVariable Long accessKeyId) {
        Long userId = (Long) request.getAttribute("userId");
        accessKeysService.revokeAccessKey(userId, accessKeyId);
        return R.success();
    }
}
