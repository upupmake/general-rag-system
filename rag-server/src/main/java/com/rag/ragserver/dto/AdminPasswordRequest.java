package com.rag.ragserver.dto;

import lombok.Data;

import javax.validation.constraints.NotBlank;

@Data
public class AdminPasswordRequest {

    @NotBlank(message = "请输入管理员密码")
    private String password;
}
