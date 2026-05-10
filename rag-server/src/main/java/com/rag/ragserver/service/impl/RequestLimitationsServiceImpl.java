package com.rag.ragserver.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.rag.ragserver.domain.RequestLimitations;
import com.rag.ragserver.service.RequestLimitationsService;
import com.rag.ragserver.mapper.RequestLimitationsMapper;
import org.springframework.stereotype.Service;

/**
* @author make
* @description 针对表【request_limitations】的数据库操作Service实现
* @createDate 2026-05-10 17:32:20
*/
@Service
public class RequestLimitationsServiceImpl extends ServiceImpl<RequestLimitationsMapper, RequestLimitations>
    implements RequestLimitationsService{

}




