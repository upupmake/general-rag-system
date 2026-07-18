package com.rag.ragserver.controller;

import com.rag.ragserver.common.R;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.servlet.http.HttpServletRequest;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/openapi/v1/auth")
public class OpenApiAuthController {
    @GetMapping("/verify")
    public R<Map<String, Object>> verify(HttpServletRequest request) {
        Map<String, Object> data = new HashMap<>();
        data.put("valid", true);
        data.put("userId", request.getAttribute("userId"));
        data.put("accessKeyId", request.getAttribute("accessKeyId"));
        return R.success(data);
    }
}
