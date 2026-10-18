package com.rag.ragserver;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.transaction.annotation.EnableTransactionManagement;

@EnableScheduling
@SpringBootApplication
@MapperScan("com.rag.ragserver.mapper")
public class RagServerApplication {

    public static void main(String[] args) {
        SpringApplication.run(RagServerApplication.class, args);
    }

}
