package com.rag.ragserver.dto;

import lombok.Data;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Size;

@Data
public class MembershipBzUpdateRequest {

    @NotBlank(message = "备注格式必须为：yyyy-MM-dd 开通N天")
    @Size(max = 250, message = "备注不能超过 250 个字符")
    private String bz;

    /**
     * 前端读取到的原备注，用于乐观并发校验；null 表示原值为 null。
     */
    private String originalBz;
}
