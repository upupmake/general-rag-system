CREATE TABLE `audit_logs` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '审计日志 ID' ,
  `user_id` BIGINT NOT NULL COMMENT '操作用户 ID' ,
  `action` VARCHAR(64) NOT NULL COMMENT '操作类型（如 CREATE_KB / QUERY / UPLOAD_DOC）' ,
  `target_type` VARCHAR(64) NOT NULL COMMENT '操作对象类型（KB / DOCUMENT / USER 等）' ,
  `target_id` BIGINT NULL COMMENT '操作对象 ID' ,
  `detail` JSON NULL COMMENT '操作详情（扩展信息）' ,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '操作时间' ,
  `status` ENUM('SUCCESS','FAIL') NOT NULL DEFAULT 'SUCCESS' ,
  `error_message` TEXT NULL,
  `duration` BIGINT NULL DEFAULT 0 ,
  `display_message` TEXT NULL,
   PRIMARY KEY (`id`)
)
ENGINE = InnoDB
COMMENT = '系统操作审计日志表';
CREATE TABLE `conversation_messages` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '对话消息 ID' ,
  `session_id` BIGINT NOT NULL COMMENT '所属会话 ID，对应 query_sessions.id' ,
  `user_id` BIGINT NOT NULL COMMENT '用户 ID（冗余字段，便于查询与审计）' ,
  `kb_id` BIGINT NULL COMMENT '当前使用的知识库 ID（冗余字段）' ,
  `role` ENUM('user','assistant','system') NOT NULL COMMENT '消息角色：user / assistant / system' ,
  `content` MEDIUMTEXT NOT NULL COMMENT '消息文本内容' ,
  `status` ENUM('pending','generating','completed','aborted','error') NOT NULL DEFAULT 'pending'  COMMENT '消息的状态 \'pending\',\'generating\',\'completed\',\'aborted\',\'error\'' ,
  `model_id` BIGINT NOT NULL COMMENT '本次生成使用的模型（assistant 消息才有）' ,
  `prompt_tokens` INT NULL COMMENT 'prompt token 数' ,
  `completion_tokens` INT NULL COMMENT 'completion token 数' ,
  `total_tokens` INT NULL COMMENT '总 token 数' ,
  `rag_context` JSON NULL COMMENT 'RAG 检索上下文信息（命中的 chunk / doc / score 等）' ,
  `latency_ms` BIGINT NULL COMMENT '本次生成耗时（毫秒）' ,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '消息创建时间' ,
  `is_deleted` INT NOT NULL DEFAULT 0 ,
  `options` JSON NULL,
  `thinking` MEDIUMTEXT NULL,
   PRIMARY KEY (`id`)
)
ENGINE = InnoDB
COMMENT = 'RAG 对话消息历史表';
CREATE TABLE `document_chunks` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '文档切片 ID' ,
  `document_id` BIGINT NOT NULL COMMENT '所属文档 ID' ,
  `kb_id` BIGINT NOT NULL COMMENT '所属知识库 ID（冗余字段，加速过滤）' ,
  `chunk_index` INT NOT NULL COMMENT '文档内切片顺序号' ,
  `text` MEDIUMTEXT NOT NULL COMMENT '切片后的文本内容' ,
  `token_length` INT NOT NULL COMMENT '该切片的 token 数' ,
  `vector_id` VARCHAR(128) NOT NULL COMMENT 'Milvus 中对应的向量 ID' ,
  `metadata` JSON NULL COMMENT '切片级元数据（页码、标题、来源等）' ,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '创建时间' ,
   PRIMARY KEY (`id`),
  CONSTRAINT `uk_doc_chunk` UNIQUE (`document_id`, `chunk_index`)
)
ENGINE = InnoDB
COMMENT = '文档切分后的最小语义单元表';
CREATE TABLE `documents` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '文档 ID' ,
  `kb_id` BIGINT NOT NULL COMMENT '所属知识库 ID' ,
  `file_path` VARCHAR(512) NOT NULL COMMENT 'MinIO 中的对象存储路径' ,
  `file_name` VARCHAR(255) NOT NULL COMMENT '原始文件名' ,
  `mime_type` VARCHAR(64) NULL DEFAULT ''  COMMENT '文件 MIME 类型' ,
  `file_size` BIGINT NOT NULL COMMENT '文件大小（字节）' ,
  `uploader_id` BIGINT NOT NULL COMMENT '上传者用户 ID' ,
  `status` ENUM('processing','ready','failed') NOT NULL DEFAULT 'processing'  COMMENT '处理状态：processing/ready/failed' ,
  `checksum` VARCHAR(64) NULL COMMENT '文件内容校验值（用于去重）' ,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '创建时间' ,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '更新时间' ,
  `is_deleted` INT NOT NULL DEFAULT 0 ,
   PRIMARY KEY (`id`)
)
ENGINE = InnoDB
COMMENT = '知识库原始文档表';
CREATE TABLE `kb_shares` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '共享记录 ID' ,
  `kb_id` BIGINT NOT NULL COMMENT '知识库 ID' ,
  `user_id` BIGINT NOT NULL COMMENT '被授权用户 ID' ,
  `granted_by` BIGINT NOT NULL COMMENT '授权人用户 ID' ,
  `granted_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '授权时间' ,
  `is_deleted` INT NOT NULL DEFAULT 0 ,
   PRIMARY KEY (`id`)
)
ENGINE = InnoDB
COMMENT = '知识库共享与权限控制表';
CREATE TABLE `knowledge_bases` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '知识库 ID' ,
  `name` VARCHAR(128) NOT NULL COMMENT '知识库名称' ,
  `owner_user_id` BIGINT NOT NULL COMMENT '知识库拥有者用户 ID' ,
  `workspace_id` BIGINT NULL COMMENT '所属工作空间 ID' ,
  `visibility` ENUM('private','shared','public') NOT NULL DEFAULT 'private'  COMMENT '可见性：private 私有，public 公共' ,
  `description` VARCHAR(255) NULL COMMENT '知识库描述' ,
  `system_prompt` MEDIUMTEXT NULL COMMENT '系统提示词' ,
  `metadata` JSON NULL COMMENT '扩展元数据（如 embedding 模型、语言等）' ,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '创建时间' ,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '更新时间' ,
  `is_deleted` INT NOT NULL DEFAULT 0 ,
   PRIMARY KEY (`id`)
)
ENGINE = InnoDB
COMMENT = '知识库主表';
CREATE TABLE `model_permissions` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '权限配置 ID' ,
  `role_id` INT NOT NULL COMMENT '角色 ID，对应 roles.id' ,
  `model_id` BIGINT NOT NULL COMMENT '模型 ID，对应 models.id' ,
  `max_tokens` INT NULL COMMENT '单次请求最大 token 数' ,
  `qps_limit` INT NULL COMMENT '每秒最大请求数' ,
  `daily_token_limit` BIGINT NULL COMMENT '每日 token 上限（NULL 表示不限）' ,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '创建时间' ,
   PRIMARY KEY (`id`),
  CONSTRAINT `uk_role_model` UNIQUE (`role_id`, `model_id`)
)
ENGINE = InnoDB
COMMENT = '角色-模型配额与限流配置表';
CREATE TABLE `models` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '模型 ID' ,
  `name` VARCHAR(64) NOT NULL COMMENT '模型名称（如 gpt-4o / deepseek-r1）' ,
  `provider` VARCHAR(64) NOT NULL COMMENT '模型提供方（openai / deepseek / local）' ,
  `max_context_tokens` INT NULL COMMENT '模型最大上下文长度' ,
  `enabled` TINYINT NOT NULL DEFAULT 1  COMMENT '模型是否启用（0 禁用，1 启用）' ,
  `metadata` JSON NULL COMMENT '模型扩展信息（如是否支持 function calling、vision 等）' ,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '创建时间' ,
  `kb_supported` INT NULL DEFAULT 1  COMMENT '该模型是否支持知识库选择' ,
   PRIMARY KEY (`id`),
  CONSTRAINT `uk_model_name` UNIQUE (`name`)
)
ENGINE = InnoDB
COMMENT = '可用大模型定义表';
CREATE TABLE `notifications` ( 
  `id` INT AUTO_INCREMENT NOT NULL COMMENT '公告id' ,
  `content` MEDIUMTEXT NOT NULL COMMENT '公告具体内容' ,
  `created_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '创建按时间' ,
  `updated_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '更新时间' ,
  `display_type` ENUM('popup','normal') NULL DEFAULT 'normal'  COMMENT '前端展示方式，popup和marquee' ,
   PRIMARY KEY (`id`)
)
ENGINE = InnoDB
COMMENT = '前端公告';
CREATE TABLE `query_sessions` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '会话 ID' ,
  `user_id` BIGINT NOT NULL COMMENT '用户 ID' ,
  `session_key` VARCHAR(64) NOT NULL DEFAULT '新的对话'  COMMENT '前端/客户端会话标识' ,
  `last_active_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '最后一次活跃时间' ,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '创建时间' ,
  `workspace_id` BIGINT NOT NULL COMMENT '工作空间 ID' ,
  `is_deleted` INT NOT NULL DEFAULT 0 ,
   PRIMARY KEY (`id`)
)
ENGINE = InnoDB
COMMENT = 'RAG 查询会话上下文表';
CREATE TABLE `roles` ( 
  `id` INT AUTO_INCREMENT NOT NULL COMMENT '角色 ID' ,
  `name` VARCHAR(32) NOT NULL COMMENT '角色名称（如 free / pro / enterprise）' ,
  `weight` INT NOT NULL DEFAULT 0  COMMENT '角色权重，数值越大权限越高' ,
  `description` VARCHAR(255) NULL COMMENT '角色描述说明' ,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '创建时间' ,
  `daily_max_tokens` INT NULL DEFAULT 1000000  COMMENT '每日最大可用token数量' ,
   PRIMARY KEY (`id`),
  CONSTRAINT `uk_roles_name` UNIQUE (`name`)
)
ENGINE = InnoDB
COMMENT = '用户角色与等级定义表.';
CREATE TABLE `users` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '用户唯一 ID' ,
  `username` VARCHAR(64) NOT NULL COMMENT '用户名（全局唯一）' ,
  `email` VARCHAR(128) NOT NULL COMMENT '用户邮箱（全局唯一）' ,
  `pwd` VARCHAR(128) NOT NULL COMMENT '用户密码' ,
  `role_id` INT NOT NULL COMMENT '用户角色 ID，对应 roles.id' ,
  `workspace_id` BIGINT NULL COMMENT '所属工作空间 ID' ,
  `status` ENUM('active','disabled') NOT NULL DEFAULT 'active'  COMMENT '用户状态：active 启用，disabled 禁用' ,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '创建时间' ,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '更新时间' ,
  `is_deleted` INT NOT NULL DEFAULT 0 ,
   PRIMARY KEY (`id`),
  CONSTRAINT `uk_users_email` UNIQUE (`email`),
  CONSTRAINT `uk_users_username` UNIQUE (`username`)
)
ENGINE = InnoDB
COMMENT = '系统用户表';
CREATE TABLE `vector_collections` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '向量集合 ID' ,
  `kb_id` BIGINT NOT NULL COMMENT '所属知识库 ID' ,
  `embedding_model` VARCHAR(128) NOT NULL COMMENT '使用的 embedding 模型名称（如 text-embedding-3-large）' ,
  `collection_name` VARCHAR(128) NOT NULL COMMENT 'Milvus 中的 collection 名称' ,
  `dim` INT NOT NULL COMMENT '向量维度' ,
  `metric_type` VARCHAR(32) NOT NULL DEFAULT 'COSINE'  COMMENT '向量距离度量方式（COSINE / L2 / IP）' ,
  `status` ENUM('active','deprecated') NOT NULL DEFAULT 'active'  COMMENT '集合状态：active 当前使用，deprecated 已废弃' ,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '创建时间' ,
  `is_deleted` INT NOT NULL DEFAULT 0 ,
   PRIMARY KEY (`id`),
  CONSTRAINT `uk_kb_collection` UNIQUE (`kb_id`, `collection_name`)
)
ENGINE = InnoDB
COMMENT = 'Milvus 向量集合与知识库映射表';
CREATE TABLE `workspace_members` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL,
  `workspace_id` BIGINT NOT NULL,
  `user_id` BIGINT NOT NULL,
  `role` ENUM('owner','member') NOT NULL DEFAULT 'member' ,
  `joined_at` DATETIME NULL DEFAULT CURRENT_TIMESTAMP ,
  `is_deleted` TINYINT NULL DEFAULT 0 ,
   PRIMARY KEY (`id`)
)
ENGINE = InnoDB;
CREATE TABLE `workspaces` ( 
  `id` BIGINT AUTO_INCREMENT NOT NULL COMMENT '工作空间 ID' ,
  `name` VARCHAR(128) NOT NULL COMMENT '工作空间名称' ,
  `owner_user_id` BIGINT NOT NULL COMMENT '工作空间拥有者用户 ID' ,
  `description` VARCHAR(255) NULL COMMENT '工作空间描述' ,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP  COMMENT '创建时间' ,
  `is_deleted` INT NOT NULL DEFAULT 0 ,
  `can_edit` INT NULL DEFAULT 1 ,
   PRIMARY KEY (`id`)
)
ENGINE = InnoDB
COMMENT = '多租户工作空间（Workspace）表';
CREATE INDEX `idx_audit_target` 
ON `audit_logs` (
  `target_type` ASC,
  `target_id` ASC
);
CREATE INDEX `idx_audit_user` 
ON `audit_logs` (
  `user_id` ASC
);
CREATE INDEX `idx_msg_kb` 
ON `conversation_messages` (
  `kb_id` ASC
);
CREATE INDEX `idx_msg_model` 
ON `conversation_messages` (
  `model_id` ASC
);
CREATE INDEX `idx_msg_session` 
ON `conversation_messages` (
  `session_id` ASC
);
CREATE INDEX `idx_msg_time` 
ON `conversation_messages` (
  `created_at` ASC
);
CREATE INDEX `idx_msg_user` 
ON `conversation_messages` (
  `user_id` ASC
);
CREATE INDEX `user_id` 
ON `conversation_messages` (
  `created_at` ASC,
  `user_id` ASC
);
CREATE INDEX `idx_chunks_kb` 
ON `document_chunks` (
  `kb_id` ASC
);
CREATE INDEX `idx_chunks_vector` 
ON `document_chunks` (
  `vector_id` ASC
);
CREATE INDEX `file_name` 
ON `documents` (
  `file_name` ASC
);
CREATE INDEX `idx_documents_kb` 
ON `documents` (
  `kb_id` ASC
);
CREATE INDEX `idx_documents_status` 
ON `documents` (
  `status` ASC
);
CREATE INDEX `idx_documents_uploader` 
ON `documents` (
  `uploader_id` ASC
);
CREATE INDEX `idx_kb_shares_kb` 
ON `kb_shares` (
  `kb_id` ASC
);
CREATE INDEX `idx_kb_shares_user` 
ON `kb_shares` (
  `user_id` ASC
);
CREATE INDEX `uk_kb_user` 
ON `kb_shares` (
  `kb_id` ASC,
  `user_id` ASC
);
CREATE INDEX `idx_kb_workspace` 
ON `knowledge_bases` (
  `workspace_id` ASC
);
CREATE INDEX `idx_kb_visibility` 
ON `knowledge_bases` (
  `visibility` ASC
);
CREATE INDEX `idx_kb_owner` 
ON `knowledge_bases` (
  `owner_user_id` ASC
);
CREATE INDEX `idx_mp_model` 
ON `model_permissions` (
  `model_id` ASC
);
CREATE INDEX `idx_mp_role` 
ON `model_permissions` (
  `role_id` ASC
);
CREATE INDEX `idx_model_enabled` 
ON `models` (
  `enabled` ASC
);
CREATE INDEX `idx_sessions_user` 
ON `query_sessions` (
  `user_id` ASC
);
CREATE INDEX `uk_session_key` 
ON `query_sessions` (
  `session_key` ASC
);
CREATE INDEX `idx_users_role` 
ON `users` (
  `role_id` ASC
);
CREATE INDEX `idx_users_workspace` 
ON `users` (
  `workspace_id` ASC
);
CREATE INDEX `idx_vector_kb` 
ON `vector_collections` (
  `kb_id` ASC
);
CREATE INDEX `idx_vector_model` 
ON `vector_collections` (
  `embedding_model` ASC
);
CREATE INDEX `workspace_id` 
ON `workspace_members` (
  `workspace_id` ASC,
  `user_id` ASC
);
CREATE INDEX `idx_workspace_owner` 
ON `workspaces` (
  `owner_user_id` ASC
);
CREATE INDEX `uk_workspace_name_owner` 
ON `workspaces` (
  `name` ASC
);
ALTER TABLE `model_permissions` ADD CONSTRAINT `model_permissions_ibfk_1` FOREIGN KEY (`model_id`) REFERENCES `models` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT;
