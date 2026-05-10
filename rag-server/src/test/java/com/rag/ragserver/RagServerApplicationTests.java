package com.rag.ragserver;

import com.rag.ragserver.domain.ConversationMessages;
import com.rag.ragserver.service.ConversationMessagesService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;

@SpringBootTest
class RagServerApplicationTests {

    @Autowired
    private ConversationMessagesService conversationMessagesService;
    @Test
    void contextLoads() {

    }

}
