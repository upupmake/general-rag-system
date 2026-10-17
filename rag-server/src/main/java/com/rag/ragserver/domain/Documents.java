package com.rag.ragserver.domain;

import com.baomidou.mybatisplus.annotation.*;

import java.util.Date;
import lombok.Data;

/**
 * 知识库原始文档表
 * @TableName documents
 */
@TableName(value ="documents")
@Data
public class Documents {
    /**
     * 文档 ID
     */
    @TableId(type = IdType.AUTO)
    private Long id;

    /**
     * 所属知识库 ID
     */
    private Long kbId;

    /**
     * MinIO 中的对象存储路径
     */
    private String filePath;

    /**
     * 原始文件名
     */
    private String fileName;

    /**
     * 文件 MIME 类型
     */
    private String mimeType;

    /**
     * 文件大小（字节）
     */
    private Long fileSize;

    /**
     * 上传者用户 ID
     */
    private Long uploaderId;

    /**
     * 处理状态：processing/ready/failed
     */
    private Object status;

    /**
     * 处理失败原因
     */
    private String errorMessage;

    /**
     * 文件内容校验值（用于去重）
     */
    private String checksum;

    /**
     * 创建时间
     */
    private Date createdAt;

    /**
     * 更新时间
     */
    private Date updatedAt;

    /**
     * 
     */
    @TableLogic
    private Integer isDeleted;

    @Override
    public boolean equals(Object that) {
        if (this == that) {
            return true;
        }
        if (that == null) {
            return false;
        }
        if (getClass() != that.getClass()) {
            return false;
        }
        Documents other = (Documents) that;
        return (this.getId() == null ? other.getId() == null : this.getId().equals(other.getId()))
            && (this.getKbId() == null ? other.getKbId() == null : this.getKbId().equals(other.getKbId()))
            && (this.getFilePath() == null ? other.getFilePath() == null : this.getFilePath().equals(other.getFilePath()))
            && (this.getFileName() == null ? other.getFileName() == null : this.getFileName().equals(other.getFileName()))
            && (this.getMimeType() == null ? other.getMimeType() == null : this.getMimeType().equals(other.getMimeType()))
            && (this.getFileSize() == null ? other.getFileSize() == null : this.getFileSize().equals(other.getFileSize()))
            && (this.getUploaderId() == null ? other.getUploaderId() == null : this.getUploaderId().equals(other.getUploaderId()))
            && (this.getStatus() == null ? other.getStatus() == null : this.getStatus().equals(other.getStatus()))
            && (this.getChecksum() == null ? other.getChecksum() == null : this.getChecksum().equals(other.getChecksum()))
            && (this.getCreatedAt() == null ? other.getCreatedAt() == null : this.getCreatedAt().equals(other.getCreatedAt()))
            && (this.getUpdatedAt() == null ? other.getUpdatedAt() == null : this.getUpdatedAt().equals(other.getUpdatedAt()))
            && (this.getIsDeleted() == null ? other.getIsDeleted() == null : this.getIsDeleted().equals(other.getIsDeleted()));
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((getId() == null) ? 0 : getId().hashCode());
        result = prime * result + ((getKbId() == null) ? 0 : getKbId().hashCode());
        result = prime * result + ((getFilePath() == null) ? 0 : getFilePath().hashCode());
        result = prime * result + ((getFileName() == null) ? 0 : getFileName().hashCode());
        result = prime * result + ((getMimeType() == null) ? 0 : getMimeType().hashCode());
        result = prime * result + ((getFileSize() == null) ? 0 : getFileSize().hashCode());
        result = prime * result + ((getUploaderId() == null) ? 0 : getUploaderId().hashCode());
        result = prime * result + ((getStatus() == null) ? 0 : getStatus().hashCode());
        result = prime * result + ((getChecksum() == null) ? 0 : getChecksum().hashCode());
        result = prime * result + ((getCreatedAt() == null) ? 0 : getCreatedAt().hashCode());
        result = prime * result + ((getUpdatedAt() == null) ? 0 : getUpdatedAt().hashCode());
        result = prime * result + ((getIsDeleted() == null) ? 0 : getIsDeleted().hashCode());
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getClass().getSimpleName());
        sb.append(" [");
        sb.append("Hash = ").append(hashCode());
        sb.append(", id=").append(id);
        sb.append(", kbId=").append(kbId);
        sb.append(", filePath=").append(filePath);
        sb.append(", fileName=").append(fileName);
        sb.append(", mimeType=").append(mimeType);
        sb.append(", fileSize=").append(fileSize);
        sb.append(", uploaderId=").append(uploaderId);
        sb.append(", status=").append(status);
        sb.append(", checksum=").append(checksum);
        sb.append(", createdAt=").append(createdAt);
        sb.append(", updatedAt=").append(updatedAt);
        sb.append(", isDeleted=").append(isDeleted);
        sb.append("]");
        return sb.toString();
    }
}