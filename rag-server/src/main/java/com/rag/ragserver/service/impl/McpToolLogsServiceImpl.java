package com.rag.ragserver.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.rag.ragserver.domain.McpToolLogs;
import com.rag.ragserver.mapper.McpToolLogsMapper;
import com.rag.ragserver.service.McpToolLogsService;
import org.springframework.stereotype.Service;

@Service
public class McpToolLogsServiceImpl extends ServiceImpl<McpToolLogsMapper, McpToolLogs>
        implements McpToolLogsService {
    @Override
    public boolean saveIfAbsent(McpToolLogs log) {
        return baseMapper.insertIgnore(log) > 0;
    }
}
