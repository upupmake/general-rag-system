package com.rag.ragserver.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.rag.ragserver.domain.McpToolLogs;

public interface McpToolLogsService extends IService<McpToolLogs> {
    boolean saveIfAbsent(McpToolLogs log);
}
