package com.rag.ragserver.controller;

import com.rag.ragserver.common.R;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Collections;
import java.util.Map;

@RestController
@RequestMapping("/openapi/v1/auth")
public class OpenApiAuthController {
    @GetMapping("/verify")
    public R<Map<String, Boolean>> verify() {
        return R.success(Collections.singletonMap("valid", true));
    }
}
