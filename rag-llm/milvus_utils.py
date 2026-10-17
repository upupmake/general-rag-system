import asyncio
import logging
import time
from typing import Dict, Optional

from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from pymilvus.client.types import LoadState

logger = logging.getLogger(__name__)


class _MilvusWrapper:
    """
    单个 collection 的运行时包装
    """

    def __init__(self, store: Milvus):
        self.store = store
        self.last_access = time.time()
        self.lock = asyncio.Lock()
        self.loaded = False
        self.load_checked_at = 0.0


class MilvusClientManager:
    """
    Milvus 连接与 collection 生命周期管理
    - 只在查询侧使用
    - 统一 load / release
    """

    _instances: Dict[str, _MilvusWrapper] = {}
    _global_lock = asyncio.Lock()

    # 空闲释放阈值（秒）
    IDLE_TTL = 30 * 60  # 30 分钟
    # 已加载状态的复用窗口，避免每次检索都做一次远程状态检查
    LOAD_STATE_TTL = 5.0

    @classmethod
    async def get_instance(
            cls,
            user_id: int,
            kb_id: int,
            milvus_uri: str,
            milvus_token: str,
            embeddings: Embeddings
    ) -> Optional[Milvus]:
        """
        获取 Milvus 实例（必要时 load collection）
        """
        db_name = f"group_{user_id // 1000}"
        collection_name = f"kb_{kb_id}"
        key = f"{db_name}.{collection_name}"

        async with cls._global_lock:
            if key not in cls._instances:
                try:
                    store = Milvus(
                        embedding_function=embeddings,
                        connection_args={
                            "uri": milvus_uri,
                            "token": milvus_token,
                            "db_name": db_name,
                        },
                        collection_name=collection_name,
                        auto_id=True,
                    )
                    cls._instances[key] = _MilvusWrapper(store)
                    logger.info(f"[Milvus] create instance: {key}")
                except Exception as e:
                    logger.error(f"[Milvus] create instance failed {key}: {e}")
                    return None

            wrapper = cls._instances[key]

        now = time.time()
        if wrapper.loaded and now - wrapper.load_checked_at < cls.LOAD_STATE_TTL:
            wrapper.last_access = now
            return wrapper.store

        # 只有一个请求负责检查/加载状态，其余并发查询在加载完成后直接放行
        async with wrapper.lock:
            now = time.time()
            if wrapper.loaded and now - wrapper.load_checked_at < cls.LOAD_STATE_TTL:
                wrapper.last_access = now
                return wrapper.store

            wrapper.last_access = now
            try:
                # pymilvus 的同步客户端会阻塞事件循环，放到线程池执行
                res = await asyncio.to_thread(
                    wrapper.store.client.get_load_state, collection_name
                )
            except Exception as e:
                logger.error(f"[Milvus] get load state failed {key}: {e}")
                return None

            state = res.get('state', LoadState.NotLoad)
            if state == LoadState.NotLoad:
                try:
                    logger.info(f"[Milvus] load collection: {key}")
                    await wrapper.store.aclient.load_collection(collection_name)
                except Exception as e:
                    logger.error(f"[Milvus] load collection failed {key}: {e}")
                    return None
                wrapper.loaded = True
            elif state == LoadState.Loaded:
                wrapper.loaded = True
            else:
                wrapper.loaded = False
            wrapper.load_checked_at = time.time()
        return wrapper.store

    @classmethod
    async def release_idle_collections(cls):
        """
        释放长时间未访问的 collection
        """
        now = time.time()
        release_keys = []

        async with cls._global_lock:
            for key, wrapper in cls._instances.items():
                if now - wrapper.last_access > cls.IDLE_TTL:
                    release_keys.append((key, wrapper))

        for key, wrapper in release_keys:
            async with wrapper.lock:
                try:
                    res = await asyncio.to_thread(
                        wrapper.store.client.get_load_state,
                        wrapper.store.collection_name,
                    )
                    state = res.get('state', LoadState.NotLoad)
                    if state == LoadState.Loaded:
                        logger.info(f"[Milvus] release collection: {key}")
                        await wrapper.store.aclient.release_collection(wrapper.store.collection_name)
                    else:
                        logger.info(f"[Milvus] collection already released: {key}")
                    wrapper.loaded = False
                except Exception as e:
                    logger.warning(f"[Milvus] release failed {key}: {e}")

            async with cls._global_lock:
                cls._instances.pop(key, None)

    @classmethod
    async def close_all(cls):
        """
        服务关闭时释放所有 collection
        """
        async with cls._global_lock:
            items = list(cls._instances.items())
            cls._instances.clear()

        for key, wrapper in items:
            async with wrapper.lock:
                try:
                    res = await asyncio.to_thread(
                        wrapper.store.client.get_load_state,
                        wrapper.store.collection_name,
                    )
                    state = res.get('state', LoadState.NotLoad)
                    if state == LoadState.Loaded:
                        logger.info(f"[Milvus] shutdown release collection: {key}")
                        await wrapper.store.aclient.release_collection(wrapper.store.collection_name)
                except Exception as e:
                    logger.warning(f"[Milvus] shutdown release failed {key}: {e}")

    @classmethod
    async def milvus_release_worker(cls):
        """
        后台定时释放 Milvus collection
        """
        while True:
            try:
                await cls.release_idle_collections()
            except Exception as e:
                logger.warning(f"[Milvus] release worker error: {e}")
            await asyncio.sleep(300)  # 每 5 分钟扫描一次
