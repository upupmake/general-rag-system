package com.rag.ragserver.task;

import com.rag.ragserver.domain.Users;
import com.rag.ragserver.service.UsersService;
import com.rag.ragserver.utils.MembershipBzUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.util.Date;
import java.util.List;

/**
 * 会员过期降级任务。
 * bz 格式固定为 "yyyy-MM-dd 开通N天"，到期日为开通日 + N 天，
 * 到期日当天仍有效，次日凌晨统一将 role_id 从 1 改为 2。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class MembershipExpireTask {

    private static final int MEMBER_ROLE_ID = 1;
    private static final int NORMAL_ROLE_ID = 2;

    private final UsersService usersService;

    @Scheduled(cron = "0 5 0 * * ?", zone = "Asia/Shanghai")
    public void downgradeExpiredMembers() {
        LocalDate today = MembershipBzUtils.today();

        List<Users> members = usersService.lambdaQuery()
                .select(Users::getId, Users::getBz)
                .eq(Users::getRoleId, MEMBER_ROLE_ID)
                .isNotNull(Users::getBz)
                .list();

        int expired = 0;
        int updated = 0;
        int invalid = 0;

        for (Users user : members) {
            MembershipBzUtils.ParsedBz parsed = MembershipBzUtils.parse(user.getBz());
            if (parsed == null) {
                invalid++;
                log.warn("会员备注格式无法解析，跳过降级, userId={}, bz={}", user.getId(), user.getBz());
                continue;
            }

            // 到期日当天仍然有效，次日凌晨才降级
            LocalDate expiryDate = parsed.getExpiryDate();
            if (!expiryDate.isBefore(today)) {
                continue;
            }

            expired++;

            try {
                boolean success = usersService.lambdaUpdate()
                        .eq(Users::getId, user.getId())
                        .eq(Users::getRoleId, MEMBER_ROLE_ID)
                        .eq(Users::getBz, user.getBz())
                        .set(Users::getRoleId, NORMAL_ROLE_ID)
                        .set(Users::getUpdatedAt, new Date())
                        .update();
                if (success) {
                    updated++;
                    log.info("会员过期已降级, userId={}, bz={}, expiryDate={}",
                            user.getId(), user.getBz(), expiryDate);
                }
            } catch (Exception e) {
                log.error("会员过期降级失败, userId={}, bz={}", user.getId(), user.getBz(), e);
            }
        }

        log.info("会员过期任务完成: scanned={}, expired={}, updated={}, invalid={}",
                members.size(), expired, updated, invalid);
    }
}
