package com.rag.ragserver.configuration;

import com.rag.ragserver.interceptor.AccessKeyInterceptor;
import com.rag.ragserver.interceptor.JwtInterceptor;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
@RequiredArgsConstructor
public class WebMvcConfig implements WebMvcConfigurer {
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")                   // 1. 允许所有的路径
                .allowedOriginPatterns("*")          // 2. 允许所有的源 (比 allowedOrigins("*") 更灵活，支持携带 Cookie)
                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS") // 3. 允许的请求方式
                .allowedHeaders("*")                 // 4. 允许所有的请求头
                .allowCredentials(true)              // 5. 允许携带凭证 (Cookie / Token)
                .maxAge(3600);                       // 6. 预检请求(OPTIONS)的缓存时间，单位秒
    }

    /**
     * 配置全局跨域 (允许所有路径跨域)
     */
    private final JwtInterceptor jwtInterceptor;
    private final AccessKeyInterceptor accessKeyInterceptor;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(jwtInterceptor)
                // 1. 拦截所有路径
                .addPathPatterns("/**")
                // 2. 放行的接口
                .excludePathPatterns(
                        "/error",           // Spring Boot 默认的错误处理路径，避免拦截器干扰错误响应
                        "/users/test",
                        "/users/send-code",
                        "/users/register",
                        "/users/send-reset-code",
                        "/users/reset-password",
                        "/users/login",
                        "/openapi/v1/**"
                );
        registry.addInterceptor(accessKeyInterceptor)
                .addPathPatterns("/openapi/v1/**");
    }
}
