-- 为文档处理失败原因增加持久化字段。已有库升级时执行一次。
ALTER TABLE `documents`
    ADD COLUMN `error_message` VARCHAR(1024) NULL COMMENT '处理失败原因' AFTER `status`;
