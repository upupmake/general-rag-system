package com.rag.ragserver.service.impl;

import com.rag.ragserver.configuration.MailConfig;
import com.rag.ragserver.service.EmailService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;

import javax.mail.MessagingException;
import javax.mail.internet.MimeMessage;

@Slf4j
@Service
@RequiredArgsConstructor
public class EmailServiceImpl implements EmailService {
    private final MailConfig mailConfig;
    private final JavaMailSender mailSender;

    @Override
    public void sendVerificationCode(String to, String code) {
        MimeMessage message = mailSender.createMimeMessage();
        try {
            MimeMessageHelper helper = new MimeMessageHelper(message, true);
            helper.setFrom(mailConfig.getUsername());
            helper.setTo(to);
            helper.setSubject("RAG系统注册验证码");

            String content = "<html>" +
                    "<head>" +
                    "<style>" +
                    "body { font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 0; }" +
                    ".container { background-color: #ffffff; width: 100%; max-width: 600px; margin: 20px auto; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }" +
                    ".header { text-align: center; padding-bottom: 20px; border-bottom: 1px solid #eee; }" +
                    ".header h2 { color: #333; }" +
                    ".content { padding: 20px 0; text-align: center; }" +
                    ".code { font-size: 24px; font-weight: bold; color: #007bff; letter-spacing: 5px; margin: 20px 0; }" +
                    ".footer { text-align: center; font-size: 12px; color: #999; border-top: 1px solid #eee; padding-top: 20px; }" +
                    "</style>" +
                    "</head>" +
                    "<body>" +
                    "<div class='container'>" +
                    "<div class='header'>" +
                    "<h2>欢迎注册 RAG 系统</h2>" +
                    "</div>" +
                    "<div class='content'>" +
                    "<p>您好！感谢您注册 RAG 系统。</p>" +
                    "<p>您的验证码如下，请在 5 分钟内完成验证：</p>" +
                    "<div class='code'>" + code + "</div>" +
                    "<p>如果这不是您的操作，请忽略此邮件。</p>" +
                    "</div>" +
                    "<div class='footer'>" +
                    "<p>此邮件由系统自动发送，请勿回复。</p>" +
                    "</div>" +
                    "</div>" +
                    "</body>" +
                    "</html>";

            helper.setText(content, true);
            mailSender.send(message);
            log.info("Verification code sent to {}", to);
        } catch (MessagingException e) {
            log.error("Failed to send email to {}", to, e);
            throw new RuntimeException("发送邮件失败", e);
        }
    }

    @Override
    public void sendResetPasswordCode(String to, String code) {
        MimeMessage message = mailSender.createMimeMessage();
        try {
            MimeMessageHelper helper = new MimeMessageHelper(message, true);
            helper.setFrom(mailConfig.getUsername());
            helper.setTo(to);
            helper.setSubject("RAG系统重置密码验证码");

            String content = "<html>" +
                    "<head>" +
                    "<style>" +
                    "body { font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 0; }" +
                    ".container { background-color: #ffffff; width: 100%; max-width: 600px; margin: 20px auto; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }" +
                    ".header { text-align: center; padding-bottom: 20px; border-bottom: 1px solid #eee; }" +
                    ".header h2 { color: #333; }" +
                    ".content { padding: 20px 0; text-align: center; }" +
                    ".code { font-size: 24px; font-weight: bold; color: #007bff; letter-spacing: 5px; margin: 20px 0; }" +
                    ".footer { text-align: center; font-size: 12px; color: #999; border-top: 1px solid #eee; padding-top: 20px; }" +
                    "</style>" +
                    "</head>" +
                    "<body>" +
                    "<div class='container'>" +
                    "<div class='header'>" +
                    "<h2>RAG 系统密码重置</h2>" +
                    "</div>" +
                    "<div class='content'>" +
                    "<p>您好！我们收到了您的密码重置请求。</p>" +
                    "<p>您的验证码如下，请在 5 分钟内完成验证：</p>" +
                    "<div class='code'>" + code + "</div>" +
                    "<p>如果这不是您的操作，请忽略此邮件并注意账号安全。</p>" +
                    "</div>" +
                    "<div class='footer'>" +
                    "<p>此邮件由系统自动发送，请勿回复。</p>" +
                    "</div>" +
                    "</div>" +
                    "</body>" +
                    "</html>";

            helper.setText(content, true);
            mailSender.send(message);
            log.info("Reset password code sent to {}", to);
        } catch (MessagingException e) {
            log.error("Failed to send reset password email to {}", to, e);
            throw new RuntimeException("发送邮件失败", e);
        }
    }
}
