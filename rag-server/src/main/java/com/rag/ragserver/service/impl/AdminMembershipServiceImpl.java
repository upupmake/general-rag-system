package com.rag.ragserver.service.impl;

import com.baomidou.mybatisplus.extension.conditions.update.LambdaUpdateChainWrapper;
import com.rag.ragserver.domain.Users;
import com.rag.ragserver.domain.admin.vo.AdminMemberVO;
import com.rag.ragserver.domain.admin.vo.AdminVerifyResponse;
import com.rag.ragserver.domain.admin.vo.MembershipGrantResultVO;
import com.rag.ragserver.dto.MembershipBzUpdateRequest;
import com.rag.ragserver.dto.MembershipGrantRequest;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.service.AdminMembershipService;
import com.rag.ragserver.service.UsersService;
import com.rag.ragserver.utils.MembershipBzUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.util.Date;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class AdminMembershipServiceImpl implements AdminMembershipService {

    private static final String TOKEN_KEY_PREFIX = "admin:membership:token:";
    private static final int MEMBER_ROLE_ID = 1;
    private static final int NORMAL_ROLE_ID = 2;

    private final UsersService usersService;
    private final StringRedisTemplate redisTemplate;

    @Value("${admin.membership.password}")
    private String adminPassword;

    @Value("${admin.membership.token-ttl-minutes:120}")
    private long tokenTtlMinutes;

    @Override
    public AdminVerifyResponse verifyPassword(String password) {
        if (adminPassword == null || adminPassword.isBlank()) {
            throw new BusinessException(503, "管理员密码未配置，请设置 ADMIN_MEMBERSHIP_PASSWORD 环境变量");
        }
        if (password == null || !adminPassword.equals(password)) {
            throw new BusinessException(403, "管理员密码错误");
        }

        String token = UUID.randomUUID().toString().replace("-", "");
        long expiresIn = tokenTtlMinutes * 60;
        redisTemplate.opsForValue().set(
                TOKEN_KEY_PREFIX + token,
                "1",
                expiresIn,
                TimeUnit.SECONDS
        );

        AdminVerifyResponse response = new AdminVerifyResponse();
        response.setToken(token);
        response.setExpiresIn(expiresIn);
        response.setExpiresAt(System.currentTimeMillis() + expiresIn * 1000);
        return response;
    }

    @Override
    public List<AdminMemberVO> listMembers(String adminToken) {
        checkToken(adminToken);
        LocalDate today = MembershipBzUtils.today();

        return usersService.lambdaQuery()
                .eq(Users::getRoleId, MEMBER_ROLE_ID)
                .orderByAsc(Users::getId)
                .list()
                .stream()
                .map(user -> toVo(user, today))
                .collect(Collectors.toList());
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public MembershipGrantResultVO grant(String adminToken, MembershipGrantRequest request) {
        checkToken(adminToken);

        String account = request.getAccount() == null ? "" : request.getAccount().trim();
        if (account.isEmpty()) {
            throw new BusinessException(400, "请输入用户名或邮箱");
        }

        Integer days = request.getDays();
        if (days == null || days < 1 || days > MembershipBzUtils.MAX_MEMBERSHIP_DAYS) {
            throw new BusinessException(400, "开通天数不合法");
        }

        List<Users> matches = usersService.lambdaQuery()
                .and(w -> w.eq(Users::getUsername, account).or().eq(Users::getEmail, account))
                .list();

        if (matches.isEmpty()) {
            throw new BusinessException(404, "未找到用户名或邮箱精准匹配的用户");
        }
        if (matches.size() > 1) {
            throw new BusinessException(400, "用户名或邮箱匹配到多个用户，请检查输入");
        }

        Users user = matches.get(0);
        Integer roleId = user.getRoleId();
        if (!Integer.valueOf(MEMBER_ROLE_ID).equals(roleId)
                && !Integer.valueOf(NORMAL_ROLE_ID).equals(roleId)) {
            throw new BusinessException(400, "暂不支持该角色用户的会员操作");
        }

        LocalDate today = MembershipBzUtils.today();
        String beforeBz = user.getBz();

        GrantPlan plan = resolveGrantPlan(roleId, beforeBz, days, today);
        String newBz = plan.getNewBz();

        Date updatedAt = new Date();
        LambdaUpdateChainWrapper<Users> update = usersService.lambdaUpdate()
                .eq(Users::getId, user.getId())
                .eq(Users::getRoleId, roleId)
                .set(Users::getRoleId, MEMBER_ROLE_ID)
                .set(Users::getBz, newBz)
                .set(Users::getUpdatedAt, updatedAt);

        if (beforeBz == null) {
            update.isNull(Users::getBz);
        } else {
            update.eq(Users::getBz, beforeBz);
        }

        if (!update.update()) {
            throw new BusinessException(409, "用户信息已被其他操作修改，请刷新后重试");
        }

        user.setRoleId(MEMBER_ROLE_ID);
        user.setBz(newBz);
        user.setUpdatedAt(updatedAt);

        MembershipGrantResultVO result = new MembershipGrantResultVO();
        result.setCaseType(plan.getCaseType());
        result.setMessage(plan.getMessage());
        result.setBeforeBz(beforeBz);
        result.setUser(toVo(user, today));
        log.info("管理员调整会员, userId={}, caseType={}, beforeBz={}, afterBz={}",
                user.getId(), plan.getCaseType(), beforeBz, newBz);
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public AdminMemberVO updateBz(String adminToken, Long userId, MembershipBzUpdateRequest request) {
        checkToken(adminToken);
        if (userId == null) {
            throw new BusinessException(400, "用户 ID 不能为空");
        }

        Users user = usersService.getById(userId);
        if (user == null || !Integer.valueOf(MEMBER_ROLE_ID).equals(user.getRoleId())) {
            throw new BusinessException(404, "会员不存在或已不是会员");
        }

        String newBz = request.getBz() == null ? null : request.getBz().trim();
        MembershipBzUtils.ParsedBz parsed = MembershipBzUtils.parse(newBz);
        if (parsed == null) {
            throw new BusinessException(400, "备注格式必须为：yyyy-MM-dd 开通N天");
        }
        newBz = MembershipBzUtils.format(parsed.getStartDate(), parsed.getDays());
        String originalBz = request.getOriginalBz();

        Date updatedAt = new Date();
        LambdaUpdateChainWrapper<Users> update = usersService.lambdaUpdate()
                .eq(Users::getId, userId)
                .eq(Users::getRoleId, MEMBER_ROLE_ID)
                .set(Users::getBz, newBz)
                .set(Users::getUpdatedAt, updatedAt);

        if (originalBz == null) {
            update.isNull(Users::getBz);
        } else {
            update.eq(Users::getBz, originalBz);
        }

        if (!update.update()) {
            throw new BusinessException(409, "备注已被其他操作修改，请刷新后重试");
        }

        user.setBz(newBz);
        user.setUpdatedAt(updatedAt);
        return toVo(user, MembershipBzUtils.today());
    }

    @Override
    public void logout(String adminToken) {
        if (adminToken != null && !adminToken.isBlank()) {
            redisTemplate.delete(TOKEN_KEY_PREFIX + adminToken);
        }
    }

    private void checkToken(String adminToken) {
        if (adminToken == null || adminToken.isBlank()
                || !Boolean.TRUE.equals(redisTemplate.hasKey(TOKEN_KEY_PREFIX + adminToken))) {
            throw new BusinessException(401, "管理员登录已失效，请重新输入密码");
        }
    }

    private AdminMemberVO toVo(Users user, LocalDate today) {
        AdminMemberVO vo = new AdminMemberVO();
        BeanUtils.copyProperties(user, vo);

        MembershipBzUtils.ParsedBz parsed = MembershipBzUtils.parse(user.getBz());
        if (parsed == null) {
            vo.setBzValid(false);
            vo.setExpired(false);
            return vo;
        }

        vo.setBzValid(true);
        vo.setExpireDate(parsed.getExpiryDate().toString());
        vo.setExpired(parsed.getExpiryDate().isBefore(today));
        return vo;
    }

    static GrantPlan resolveGrantPlan(int roleId, String bz, int days, LocalDate today) {
        MembershipBzUtils.ParsedBz parsed = MembershipBzUtils.parse(bz);

        if (roleId == NORMAL_ROLE_ID) {
            return new GrantPlan(
                    "ACTIVATE",
                    "普通用户已开通 " + days + " 天会员",
                    MembershipBzUtils.format(today, days)
            );
        }

        if (parsed == null) {
            return new GrantPlan(
                    "RESET",
                    "会员备注格式异常，已按新开通 " + days + " 天处理",
                    MembershipBzUtils.format(today, days)
            );
        }

        if (parsed.getExpiryDate().isBefore(today)) {
            return new GrantPlan(
                    "REACTIVATE",
                    "会员已过期，已按今天重新开通 " + days + " 天",
                    MembershipBzUtils.format(today, days)
            );
        }

        long totalDays = parsed.getDays() + days;
        if (totalDays > MembershipBzUtils.MAX_MEMBERSHIP_DAYS) {
            throw new BusinessException(400, "续费后总开通天数超过上限");
        }

        return new GrantPlan(
                "RENEW",
                "会员未过期，已续费 " + days + " 天，累计 " + totalDays + " 天",
                MembershipBzUtils.format(parsed.getStartDate(), totalDays)
        );
    }

    static final class GrantPlan {
        private final String caseType;
        private final String message;
        private final String newBz;

        GrantPlan(String caseType, String message, String newBz) {
            this.caseType = caseType;
            this.message = message;
            this.newBz = newBz;
        }

        String getCaseType() {
            return caseType;
        }

        String getMessage() {
            return message;
        }

        String getNewBz() {
            return newBz;
        }
    }
}
