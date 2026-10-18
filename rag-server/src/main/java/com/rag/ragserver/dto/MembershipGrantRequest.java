package com.rag.ragserver.dto;

import lombok.Data;

import javax.validation.constraints.Max;
import javax.validation.constraints.Min;
import javax.validation.constraints.NotBlank;
import javax.validation.constraints.NotNull;

@Data
public class MembershipGrantRequest {

    @NotBlank(message = "请输入用户名或邮箱")
    private String account;

    @NotNull(message = "请输入开通天数")
    @Min(value = 1, message = "开通天数必须大于 0")
    @Max(value = 36500, message = "开通天数不能超过 36500 天")
    private Integer days;
}
