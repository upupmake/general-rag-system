package com.rag.ragserver.domain.admin.vo;

import lombok.Data;

/**
 * 管理员会员列表返回对象。
 * 不返回 pwd 字段，避免密码泄露。
 */
@Data
public class AdminMemberVO {

    private Long id;
    private String username;
    private String email;
    private Integer roleId;
    private String bz;
    private Object status;

    /**
     * bz 是否为合法格式。
     */
    private Boolean bzValid;

    /**
     * 到期日，格式 yyyy-MM-dd；bz 非法时为空。
     */
    private String expireDate;

    /**
     * 是否已过期（到期日当天不算过期）。
     */
    private Boolean expired;
}
