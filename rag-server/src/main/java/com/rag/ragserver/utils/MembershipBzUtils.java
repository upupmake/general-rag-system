package com.rag.ragserver.utils;

import java.time.DateTimeException;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 会员备注 bz 的统一解析与格式化。
 * 固定格式："yyyy-MM-dd 开通N天"，到期日为开通日 + N 天。
 */
public final class MembershipBzUtils {

    public static final ZoneId ZONE = ZoneId.of("Asia/Shanghai");
    public static final long MAX_MEMBERSHIP_DAYS = 36500L;

    private static final Pattern BZ_PATTERN =
            Pattern.compile("^([0-9]{4}-[0-9]{2}-[0-9]{2})[\\s\\u3000]+开通([0-9]+)天$");
    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ISO_LOCAL_DATE;

    private MembershipBzUtils() {
    }

    public static LocalDate today() {
        return LocalDate.now(ZONE);
    }

    public static ParsedBz parse(String bz) {
        if (bz == null) {
            return null;
        }

        String value = bz.strip();
        if (value.isEmpty()) {
            return null;
        }

        Matcher matcher = BZ_PATTERN.matcher(value);
        if (!matcher.matches()) {
            return null;
        }

        try {
            long days = Long.parseLong(matcher.group(2));
            if (days <= 0 || days > MAX_MEMBERSHIP_DAYS) {
                return null;
            }
            return new ParsedBz(LocalDate.parse(matcher.group(1)), days);
        } catch (DateTimeException | NumberFormatException e) {
            return null;
        }
    }

    public static String format(LocalDate startDate, long days) {
        return startDate.format(DATE_FORMATTER) + " 开通" + days + "天";
    }

    public static final class ParsedBz {
        private final LocalDate startDate;
        private final long days;

        public ParsedBz(LocalDate startDate, long days) {
            this.startDate = startDate;
            this.days = days;
        }

        public LocalDate getStartDate() {
            return startDate;
        }

        public long getDays() {
            return days;
        }

        public LocalDate getExpiryDate() {
            return startDate.plusDays(days);
        }
    }
}
