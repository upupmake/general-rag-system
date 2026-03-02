package com.rag.ragserver.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.rag.ragserver.common.R;
import com.rag.ragserver.domain.Roles;
import com.rag.ragserver.domain.Users;
import com.rag.ragserver.domain.Workspaces;
import com.rag.ragserver.dto.LoginRequest;
import com.rag.ragserver.exception.BusinessException;
import com.rag.ragserver.service.RolesService;
import com.rag.ragserver.service.UsersService;
import com.rag.ragserver.service.WorkspacesService;
import com.rag.ragserver.utils.JwtUtils;
import io.jsonwebtoken.Claims;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.web.bind.annotation.*;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import com.rag.ragserver.dto.RegisterRequest;
import com.rag.ragserver.dto.SendCodeRequest;
import com.rag.ragserver.service.EmailService;

import javax.validation.Valid;

@Slf4j
@RestController
@RequestMapping("/users")
@RequiredArgsConstructor
public class UserController {
    private final UsersService usersService;
    private final RolesService rolesService;
    private final WorkspacesService workspacesService;
    private final JwtUtils jwtUtils;
    private final StringRedisTemplate redisTemplate;
    private final EmailService emailService;

    @GetMapping("/test")
    public R<String> test() {
        return R.success("UserController is working!");
    }

    @PostMapping("/send-code")
    public R<String> sendCode(@RequestBody @Valid SendCodeRequest request) {
        String email = request.getEmail();
        // Check if email already registered
        long count = usersService.lambdaQuery().eq(Users::getEmail, email).count();
        if (count > 0) {
            throw new BusinessException(400, "该邮箱已被注册");
        }

        String code = String.valueOf((int) ((Math.random() * 9 + 1) * 100000));
        redisTemplate.opsForValue().set("register:code:" + email, code, 5, TimeUnit.MINUTES);

        emailService.sendVerificationCode(email, code);
        return R.success("验证码已发送");
    }

    @PostMapping("/register")
    public R<String> register(@RequestBody @Valid RegisterRequest request) {
        String email = request.getEmail();
        String code = request.getCode();

        String cacheCode = redisTemplate.opsForValue().get("register:code:" + email);
        if (cacheCode == null || !cacheCode.equals(code)) {
            throw new BusinessException(400, "验证码错误或已失效");
        }

        // Double check username and email
        if (usersService.lambdaQuery().eq(Users::getUsername, request.getUsername()).exists()) {
            throw new BusinessException(400, "用户名已存在");
        }
        if (usersService.lambdaQuery().eq(Users::getEmail, email).exists()) {
            throw new BusinessException(400, "邮箱已存在");
        }

        Users user = new Users();
        user.setUsername(request.getUsername());
        user.setPwd(request.getPassword()); // In real app, password should be encrypted
        user.setEmail(email);
        // Check if email domain is allowed; disable account if not
        List<String> allowedDomains = Arrays.asList(
                "qq.com", "foxmail.com", "163.com", "126.com", "yeah.net",
                "sina.com", "sina.cn", "sohu.com", "139.com", "189.cn",
                "gmail.com", "outlook.com", "hotmail.com", "live.com", "msn.com",
                "yahoo.com", "icloud.com", "aol.com", "gmx.com", "proton.me"
        );
        String emailLower = email.toLowerCase();
        boolean domainAllowed = allowedDomains.stream().anyMatch(domain -> emailLower.endsWith("@" + domain))
                || emailLower.endsWith(".edu.cn") || emailLower.endsWith(".edu");
        user.setStatus(domainAllowed ? "active" : "disabled");
        user.setRoleId(2); // Default role, assuming 2 is user

        usersService.save(user);

        // Initialize workspace
        workspacesService.validateAndFixUserWorkspace(user.getId());

        redisTemplate.delete("register:code:" + email);

        return R.success("注册成功");
    }

    @PostMapping("/login")
    public R<Map<String, Object>> login(@RequestBody LoginRequest request) {

        LambdaQueryWrapper<Users> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.and(w -> w.eq(Users::getUsername, request.getUsername())
                        .or()
                        .eq(Users::getEmail, request.getUsername()))
                .eq(Users::getPwd, request.getPassword());

        Users user = usersService.getOne(queryWrapper);

        if (user == null) {
            throw new BusinessException(400, "用户名或密码错误");
        }

        if (!"active".equals(user.getStatus())) {
            throw new BusinessException(403, "用户已被禁用");
        }

        workspacesService.validateAndFixUserWorkspace(user.getId());

        String jti = UUID.randomUUID().toString();
        boolean rememberMe = Boolean.TRUE.equals(request.getRememberMe());
        String token = jwtUtils.generateToken(
                user.getId(),
                user.getUsername(),
                rememberMe,
                jti
        );

        long expiration = jwtUtils.getExpirationTime(rememberMe);
        redisTemplate.opsForValue().set("login:token:" + jti, user.getId().toString(), expiration, TimeUnit.MILLISECONDS);

        Map<String, Object> userInfo = new HashMap<>();
        userInfo.put("id", user.getId());
        userInfo.put("username", user.getUsername());
        userInfo.put("email", user.getEmail());

        Roles role = rolesService.getById(user.getRoleId());
        if (role != null) {
            userInfo.put("role", Map.of(
                    "id", role.getId(),
                    "name", role.getName(),
                    "weight", role.getWeight()
            ));
        }
        return R.success(Map.of(
                "token", token,
                "user", userInfo
        ));
    }

    @PostMapping("/logout")
    public R<String> logout(@RequestHeader("Authorization") String authHeader) {
        if (authHeader == null || authHeader.isEmpty()) {
            throw new BusinessException(401, "未登录");
        }
        String token = authHeader;
        if (token.startsWith("Bearer ")) {
            token = token.substring(7);
        }
        try {
            Claims claims = jwtUtils.extractAllClaims(token);
            String jti = claims.getId();

            if (jti != null) {
                String redisKey = "login:token:" + jti;
                Boolean deleted = redisTemplate.delete(redisKey);
                if (!Boolean.TRUE.equals(deleted)) {
                    throw new BusinessException(400, "Token不存在或已失效");
                }
            }
            return R.success("退出成功");
        } catch (Exception e) {
            throw new BusinessException(401, "Token无效");
        }
    }
}
