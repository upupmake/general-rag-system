-- 加速上传同名校验：按知识库 + 完整文件路径查询。
ALTER TABLE `documents`
    ADD INDEX `idx_documents_kb_file_name` (`kb_id`, `file_name`);
