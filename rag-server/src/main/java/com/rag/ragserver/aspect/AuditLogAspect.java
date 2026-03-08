package com.rag.ragserver.aspect;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.rag.ragserver.domain.AuditLogs;
import com.rag.ragserver.service.AuditLogsService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import org.springframework.web.multipart.MultipartFile;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.Date;
import java.util.Arrays;

@Aspect
@Component
@Slf4j
@RequiredArgsConstructor
public class AuditLogAspect {

    private final AuditLogsService auditLogsService;
    private final ObjectMapper objectMapper;

    // Pointcut targeting all public methods in controllers
    @Pointcut("execution(* com.rag.ragserver.controller..*.*(..))")
    public void controllerMethods() {
    }

    @Around("controllerMethods()")
    public Object logAudit(ProceedingJoinPoint joinPoint) throws Throwable {
        long startTime = System.currentTimeMillis();
        Object result = null;
        String errorMessage = null;
        String status = "SUCCESS";

        try {
            result = joinPoint.proceed();
            return result;
        } catch (Throwable e) {
            status = "FAIL";
            errorMessage = e.getMessage();
            throw e;
        } finally {
            try {
                long duration = System.currentTimeMillis() - startTime;
                saveAuditLog(joinPoint, status, errorMessage, duration);
            } catch (Exception e) {
                log.error("Failed to save audit log", e);
            }
        }
    }

    private void saveAuditLog(ProceedingJoinPoint joinPoint, String status, String errorMessage, long duration) {
        ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (attributes == null) {
            return;
        }
        HttpServletRequest request = attributes.getRequest();

        AuditLogs auditLog = new AuditLogs();
        
        // 1. User Info
        Long userId = (Long) request.getAttribute("userId");
        auditLog.setUserId(userId != null ? userId : 0L); // 0 for anonymous/system

        // 2. Action & Target
        String className = joinPoint.getTarget().getClass().getSimpleName();
        String methodName = joinPoint.getSignature().getName();
        auditLog.setAction(methodName);
        auditLog.setTargetType(className.replace("Controller", ""));
        auditLog.setTargetId(null); 

        
        // 3. Details (Args)
        try {
             Object[] args = joinPoint.getArgs();
             if (args != null && args.length > 0) {
                 // Filter out request/response/multipart objects
                 Object[] printableArgs = Arrays.stream(args)
                     .filter(arg -> !(arg instanceof javax.servlet.ServletRequest) 
                                 && !(arg instanceof javax.servlet.ServletResponse)
                                 && !(arg instanceof MultipartFile)
                                 && !(arg instanceof MultipartFile[]))
                     .toArray();
                 
                 // Always use JSON serialization to ensure valid JSON format for the database column
                 if (printableArgs.length == 1) {
                     auditLog.setDetail(objectMapper.writeValueAsString(printableArgs[0]));
                 } else {
                     auditLog.setDetail(objectMapper.writeValueAsString(printableArgs));
                 }
             }
        } catch (Exception e) {
            // log.warn("Audit log serialization failed: {}", e.getMessage());
            auditLog.setDetail("Args serialization failed");
        }

        // 5. Status & Timing
        auditLog.setStatus(status);
        auditLog.setErrorMessage(errorMessage);
        auditLog.setDuration(duration);
        auditLog.setCreatedAt(new Date());

        // 6. Generate Display Message
        auditLog.setDisplayMessage(generateDisplayMessage(methodName));

        log.info("[{}] {} - {}ms", request.getMethod(), request.getRequestURI(), duration);
        auditLogsService.save(auditLog);
    }

    private String generateDisplayMessage(String action) {
        switch (action) {
            case "createKnowledgeBase": return "创建了知识库";
            case "deleteKnowledgeBase": return "删除了知识库";
            case "uploadDocuments": return "上传了文档";
            case "deleteDocument": return "删除了文档";
            case "renameDocument": return "重命名了文档";
            case "inviteUserToKb": return "邀请用户加入知识库";
            case "removeUserFromKb": return "移除了知识库成员";
            case "createWorkspace": return "创建了工作空间";
            case "inviteUserToWorkspace": return "邀请用户加入工作空间";
            case "switchWorkspace": return "切换了工作空间";
            case "removeUserFromWorkspace": return "移除了工作空间成员";
            default: return action;
        }
    }
}
