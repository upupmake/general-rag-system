package com.rag.ragserver.dto;

import lombok.Data;
import org.hibernate.validator.constraints.Length;

import javax.validation.constraints.NotBlank;

@Data
public class AccessKeyCreateDTO {
    @NotBlank(message = "Access Key 名称不能为空")
    @Length(max = 100, message = "Access Key 名称不能超过100个字符")
    private String name;
}
