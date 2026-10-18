package com.rag.ragserver.service.impl;

import com.rag.ragserver.exception.BusinessException;
import org.junit.jupiter.api.Test;

import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class AdminMembershipServiceImplTest {

    private static final LocalDate TODAY = LocalDate.of(2026, 6, 9);

    @Test
    void activatesNormalUserFromToday() {
        AdminMembershipServiceImpl.GrantPlan plan =
                AdminMembershipServiceImpl.resolveGrantPlan(2, "旧备注", 30, TODAY);

        assertEquals("ACTIVATE", plan.getCaseType());
        assertEquals("2026-06-09 开通30天", plan.getNewBz());
    }

    @Test
    void reactivatesExpiredMemberFromToday() {
        AdminMembershipServiceImpl.GrantPlan plan =
                AdminMembershipServiceImpl.resolveGrantPlan(1, "2026-05-01 开通5天", 10, TODAY);

        assertEquals("REACTIVATE", plan.getCaseType());
        assertEquals("2026-06-09 开通10天", plan.getNewBz());
    }

    @Test
    void renewsActiveMemberKeepingStartDate() {
        AdminMembershipServiceImpl.GrantPlan plan =
                AdminMembershipServiceImpl.resolveGrantPlan(1, "2026-06-01 开通30天", 10, TODAY);

        assertEquals("RENEW", plan.getCaseType());
        assertEquals("2026-06-01 开通40天", plan.getNewBz());
    }

    @Test
    void treatsExpiryDayAsStillActive() {
        AdminMembershipServiceImpl.GrantPlan plan =
                AdminMembershipServiceImpl.resolveGrantPlan(1, "2026-06-01 开通8天", 2, TODAY);

        assertEquals("RENEW", plan.getCaseType());
        assertEquals("2026-06-01 开通10天", plan.getNewBz());
    }

    @Test
    void resetsMalformedRemark() {
        AdminMembershipServiceImpl.GrantPlan plan =
                AdminMembershipServiceImpl.resolveGrantPlan(1, "不是标准格式", 10, TODAY);

        assertEquals("RESET", plan.getCaseType());
        assertEquals("2026-06-09 开通10天", plan.getNewBz());
    }

    @Test
    void rejectsRenewalOverMaximumDays() {
        BusinessException exception = assertThrows(BusinessException.class,
                () -> AdminMembershipServiceImpl.resolveGrantPlan(
                        1, "2026-06-01 开通36500天", 1, TODAY));

        assertEquals(400, exception.getCode());
    }
}
