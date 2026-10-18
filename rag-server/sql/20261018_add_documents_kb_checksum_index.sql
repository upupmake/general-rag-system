-- 加速上传去重查询：按知识库 + 内容 MD5 查找同目录重复文件。
-- 索引非唯一，目录级唯一性由应用层判断。
ALTER TABLE `documents`
    ADD INDEX `idx_documents_kb_checksum` (`kb_id`, `checksum`);
