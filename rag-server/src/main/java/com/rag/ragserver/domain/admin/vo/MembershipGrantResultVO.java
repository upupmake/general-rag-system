package com.rag.ragserver.domain.admin.vo;

import lombok.Data;

@Data
public class MembershipGrantResultVO {

    /**
     * ACTIVATE：普通用户开通；REACTIVATE：过期会员重新开通；
     * RENEW：未过期会员续费；RESET：备注异常按新开通处理。
     */
    private String caseType;

    private String message;

    private String beforeBz;

    private AdminMemberVO user;
}
