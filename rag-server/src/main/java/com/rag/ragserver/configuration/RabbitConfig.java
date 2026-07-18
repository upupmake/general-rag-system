package com.rag.ragserver.configuration;

import org.springframework.amqp.core.*;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.amqp.support.converter.MessageConverter;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.config.SimpleRabbitListenerContainerFactory;
import org.springframework.amqp.rabbit.listener.ConditionalRejectingErrorHandler;
import org.springframework.amqp.rabbit.retry.RejectAndDontRequeueRecoverer;
import org.springframework.amqp.rabbit.config.RetryInterceptorBuilder;

@Configuration
public class RabbitConfig {
    @Bean
    public MessageConverter jackson2JsonMessageConverter() {
        return new Jackson2JsonMessageConverter();
    }

    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory, MessageConverter messageConverter) {
        final RabbitTemplate rabbitTemplate = new RabbitTemplate(connectionFactory);
        //设置Json转换器
        rabbitTemplate.setMessageConverter(messageConverter);
        return rabbitTemplate;
    }

    // --- 声明生产者相关的交换机、队列以及绑定 ---
    // --- 消费者相关的使用注解生成 ---
    @Bean
    public DirectExchange serverInteractLLMExchange() {
        return new DirectExchange("server.interact.llm.exchange", true, false);
    }

    @Bean
    public Queue documentEmbeddingQueue() {
        return QueueBuilder.durable("rag.document.process.queue").build();
    }

    @Bean
    public Binding documentEmbeddingBinding(DirectExchange serverInteractLLMExchange, Queue documentEmbeddingQueue) {
        return BindingBuilder.bind(documentEmbeddingQueue)
                .to(serverInteractLLMExchange)
                .with("rag.document.process.key");
    }

    @Bean
    public DirectExchange ragAuditExchange() {
        return new DirectExchange("rag.audit.exchange", true, false);
    }

    @Bean
    public DirectExchange ragAuditDeadLetterExchange() {
        return new DirectExchange("rag.audit.dlx", true, false);
    }

    @Bean
    public Queue ragMcpToolLogQueue() {
        return QueueBuilder.durable("rag.mcp.tool.log.queue")
                .deadLetterExchange("rag.audit.dlx")
                .deadLetterRoutingKey("rag.mcp.tool.log.dead")
                .build();
    }

    @Bean
    public Queue ragMcpToolLogDeadQueue() {
        return QueueBuilder.durable("rag.mcp.tool.log.dead.queue").build();
    }

    @Bean
    public Binding ragMcpToolLogBinding(DirectExchange ragAuditExchange, Queue ragMcpToolLogQueue) {
        return BindingBuilder.bind(ragMcpToolLogQueue)
                .to(ragAuditExchange)
                .with("rag.mcp.tool.log.v1");
    }

    @Bean
    public Binding ragMcpToolLogDeadBinding(DirectExchange ragAuditDeadLetterExchange,
                                             Queue ragMcpToolLogDeadQueue) {
        return BindingBuilder.bind(ragMcpToolLogDeadQueue)
                .to(ragAuditDeadLetterExchange)
                .with("rag.mcp.tool.log.dead");
    }

    @Bean
    public SimpleRabbitListenerContainerFactory auditRabbitListenerContainerFactory(
            ConnectionFactory connectionFactory, MessageConverter messageConverter) {
        SimpleRabbitListenerContainerFactory factory = new SimpleRabbitListenerContainerFactory();
        factory.setConnectionFactory(connectionFactory);
        factory.setMessageConverter(messageConverter);
        factory.setDefaultRequeueRejected(false);
        factory.setErrorHandler(new ConditionalRejectingErrorHandler());
        factory.setAdviceChain(RetryInterceptorBuilder.stateless()
                .maxAttempts(3)
                .recoverer(new RejectAndDontRequeueRecoverer())
                .build());
        return factory;
    }
}
