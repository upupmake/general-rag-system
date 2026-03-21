package com.rag.ragserver.service;

public interface EmailService {
    void sendVerificationCode(String to, String code);

    void sendResetPasswordCode(String to, String code);
}
