package com.rag.ragserver.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.rag.ragserver.domain.McpToolLogs;
import org.apache.ibatis.annotations.Param;

public interface McpToolLogsMapper extends BaseMapper<McpToolLogs> {
    int insertIgnore(@Param("log") McpToolLogs log);
}
