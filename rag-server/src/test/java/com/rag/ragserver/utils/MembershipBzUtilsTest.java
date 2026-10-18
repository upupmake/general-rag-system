package com.rag.ragserver.utils;

import org.junit.jupiter.api.Test;

import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class MembershipBzUtilsTest {

    @Test
    void parsesExpiryDateAsStartPlusDays() {
        MembershipBzUtils.ParsedBz parsed = MembershipBzUtils.parse("2026-06-01 开通7天");

        assertEquals(LocalDate.of(2026, 6, 1), parsed.getStartDate());
        assertEquals(7, parsed.getDays());
        assertEquals(LocalDate.of(2026, 6, 8), parsed.getExpiryDate());
    }

    @Test
    void handlesLeapYearAndMonthEnd() {
        assertEquals(LocalDate.of(2024, 3, 1),
                MembershipBzUtils.parse("2024-02-01 开通29天").getExpiryDate());
        assertEquals(LocalDate.of(2026, 2, 1),
                MembershipBzUtils.parse("2026-01-31 开通1天").getExpiryDate());
    }

    @Test
    void acceptsUnicodeSpacesAndTrims() {
        assertEquals(LocalDate.of(2026, 6, 8),
                MembershipBzUtils.parse(" 2026-06-01\u3000开通7天 ").getExpiryDate());
    }

    @Test
    void formatsBzConsistently() {
        assertEquals("2026-06-01 开通7天",
                MembershipBzUtils.format(LocalDate.of(2026, 6, 1), 7));
    }

    @Test
    void returnsNullForInvalidRemarks() {
        assertNull(MembershipBzUtils.parse(null));
        assertNull(MembershipBzUtils.parse(""));
        assertNull(MembershipBzUtils.parse("2026/06/01 开通7天"));
        assertNull(MembershipBzUtils.parse("2026-06-01 开通0天"));
        assertNull(MembershipBzUtils.parse("2026-06-01 开通7天abc"));
        assertNull(MembershipBzUtils.parse("2026-02-30 开通7天"));
    }
}
