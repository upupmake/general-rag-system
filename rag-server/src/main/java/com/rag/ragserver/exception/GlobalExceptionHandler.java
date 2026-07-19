package com.rag.ragserver.exception;

import lombok.extern.slf4j.Slf4j;
import org.springframework.validation.BindException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import javax.servlet.http.HttpServletResponse;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    /**
     * 捕获自定义业务异常
     */
    @ExceptionHandler(BusinessException.class)
    public Map<String, Object> handleBusinessException(BusinessException e, HttpServletResponse response) {
        log.warn("业务异常: {}", e.getMessage());
        response.setStatus(e.getCode());
        return buildResponse(e.getCode(), e.getMessage());
    }

    /**
     * 捕获参数校验异常
     */
    @ExceptionHandler({MethodArgumentNotValidException.class, BindException.class})
    public Map<String, Object> handleValidationException(Exception e) {
        String msg = null;
        if (e instanceof MethodArgumentNotValidException) {
            msg = ((MethodArgumentNotValidException) e).getBindingResult().getAllErrors().get(0).getDefaultMessage();
        } else if (e instanceof BindException) {
            msg = ((BindException) e).getBindingResult().getAllErrors().get(0).getDefaultMessage();
        }
        log.warn("参数校验异常: {}", msg);
        return buildResponse(400, msg);
    }

    /**
     * 捕获运行时异常 (兼容你在 UserController 中直接抛出的 RuntimeException)
     */
    @ExceptionHandler(RuntimeException.class)
    public Map<String, Object> handleRuntimeException(RuntimeException e) {
        log.error("运行时异常:", e);
        // 统一返回通用错误提示，避免泄露系统日志
        return buildResponse(500, "系统内部错误，请联系管理员");
    }

    /**
     * 捕获所有其他未处理的异常
     */
    @ExceptionHandler(Exception.class)
    public Map<String, Object> handleException(Exception e) {
        log.error("系统未知异常:", e);
        return buildResponse(500, "系统繁忙，请稍后再试");
    }

    /**
     * 构造统一返回格式
     */
    private Map<String, Object> buildResponse(Integer code, String message) {
        Map<String, Object> map = new HashMap<>();
        map.put("code", code);
        map.put("message", message); // 保持与 UserController 中的 key 一致
        map.put("data", null);
        return map;
    }
}
