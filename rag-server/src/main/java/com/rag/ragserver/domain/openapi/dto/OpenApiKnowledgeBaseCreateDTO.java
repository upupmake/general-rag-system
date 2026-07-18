package com.rag.ragserver.domain.openapi.dto;

import lombok.Data;
import org.hibernate.validator.constraints.Length;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Pattern;

@Data
public class OpenApiKnowledgeBaseCreateDTO {
    @NotBlank(message = "知识库名称不能为空")
    @Length(max = 100, message = "知识库名称不能超过100个字符")
    private String name;

    @Length(max = 200, message = "知识库描述不能超过200个字符")
    private String description;

    @NotBlank(message = "知识库可见性不能为空")
    @Pattern(regexp = "private|public", message = "目前仅支持创建私有或公开知识库")
    private String visibility;
}
