package com.rag.ragserver.mapper;

import com.rag.ragserver.domain.Documents;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

/**
 * @author make
* @description 针对表【documents(知识库原始文档表)】的数据库操作Mapper
* @createDate 2025-12-31 01:13:35
* @Entity com.rag.ragserver.domain.Documents
*/
public interface DocumentsMapper extends BaseMapper<Documents> {

    @Select("SELECT COUNT(*) FROM documents d " +
            "INNER JOIN knowledge_bases kb ON kb.id = d.kb_id " +
            "WHERE d.uploader_id = #{uploaderId} " +
            "AND d.is_deleted = 0 AND kb.is_deleted = 0")
    Long countActiveByUploaderId(@Param("uploaderId") Long uploaderId);

}



