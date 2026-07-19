package com.rag.ragserver.service;

import com.rag.ragserver.domain.Documents;
import com.baomidou.mybatisplus.extension.service.IService;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletResponse;
import java.util.List;

/**
* @author make
* @description 针对表【documents(知识库原始文档表)】的数据库操作Service
* @createDate 2025-12-31 01:13:35
*/
public interface DocumentsService extends IService<Documents> {

    List<Documents> listByKbId(Long kbId);

    Long countActiveByUploaderId(Long uploaderId);

    List<Documents> uploadDocuments(Long kbId, MultipartFile[] files, Long userId);

    void deleteDocument(Long kbId, Long docId, Long userId);

    void previewDocument(Long docId, Long userId, HttpServletResponse response);
}
