-- 扩展文档 MIME 类型字段长度，容纳 .docx 等较长的标准 MIME 类型。已有库升级时执行一次。
ALTER TABLE `documents`
    MODIFY COLUMN `mime_type` VARCHAR(255) NULL DEFAULT '' COMMENT '文件 MIME 类型';
