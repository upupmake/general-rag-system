import asyncio
import io
import json
import logging
import os
import traceback

from aio_pika.abc import AbstractIncomingMessage
from langchain_core.documents import Document

import utils
from milvus_utils import MilvusClientManager
from minio_utils import minio_client
from mq.connection import rabbit_async_client
from utils import get_embedding_instance

# Configure logging
logger = logging.getLogger(__name__)

IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


class DocumentEmbeddingConsumer:
    async def error_message_sender(self, document_id: int, error_msg: str):
        response_message = {
            "documentId": document_id,
            "status": "failed",
            "message": error_msg
        }
        await rabbit_async_client.publish(
            exchange_name="server.interact.llm.exchange",
            routing_key="rag.document.complete.key",
            message=response_message
        )

    @staticmethod
    def _load_existing_vector_ids(vector_store, document_id: int) -> list[int]:
        """查询文档已存在的向量主键，用于重跑时清理旧切片。"""
        existing_ids = []
        offset = 0
        page_size = 1000
        while True:
            rows = vector_store.client.query(
                collection_name=vector_store.collection_name,
                filter=f"documentId == {document_id}",
                output_fields=["pk"],
                limit=page_size,
                offset=offset,
            )
            if not rows:
                break
            for row in rows:
                try:
                    existing_ids.append(int(row["pk"]))
                except (KeyError, TypeError, ValueError):
                    continue
            if len(rows) < page_size:
                break
            offset += len(rows)
        return existing_ids

    @staticmethod
    def _delete_vector_ids(vector_store, vector_ids: list) -> None:
        """按主键批量删除向量，单批最多 1000 条。"""
        for start in range(0, len(vector_ids), 1000):
            vector_store.client.delete(
                collection_name=vector_store.collection_name,
                ids=list(vector_ids[start:start + 1000]),
            )

    async def on_receive_message(self, message: AbstractIncomingMessage):
        async with message.process(requeue=True, reject_on_redelivered=True):
            document_id = None
            try:
                body = message.body.decode()
                logger.info(f"Received document processing task: {body}")
                data = json.loads(body)

                document_id = data.get("documentId")
                kb_id = data.get("kbId")
                user_id = data.get("userId")
                file_path = data.get("filePath")
                file_name = data.get("fileName")
                bucket_name = data.get("bucketName")
                if document_id is None:
                    raise ValueError("文档任务缺少 documentId")

                # Download file
                response = await minio_client.get_object(bucket_name, file_path)
                if not response:
                    raise Exception("Failed to download file from MinIO")

                tmp_path = os.path.join('./temp', file_path)
                parent_dir = os.path.dirname(tmp_path)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)

                suffix = os.path.splitext(file_path)[1]

                minio_byte = await response.read()

                # Only write to disk if not an image (images processed in-memory)
                is_image = suffix.lower() in IMAGE_MIME_TYPES
                if not is_image:
                    def write_temp_file():
                        with open(tmp_path, "wb") as f:
                            f.write(minio_byte)

                    await asyncio.to_thread(write_temp_file)

                vector_store = None
                try:
                    if suffix.lower() == ".pdf":
                        # chunk_overlap=0 可确保不重复
                        texts = await asyncio.to_thread(utils.pdf_split, tmp_path)
                        splits = [Document(page_content=t) for t in texts]
                    elif suffix.lower() == ".txt":
                        def split_txt():
                            with open(tmp_path, "r", encoding="utf-8") as f:
                                return [Document(page_content=t) for t in utils.plain_text_split(f.read())]

                        splits = await asyncio.to_thread(split_txt)
                    elif suffix.lower() == ".md":
                        with open(tmp_path, "r", encoding="utf-8") as f:
                            splits = await asyncio.to_thread(utils.markdown_split, f.read())
                    elif suffix.lower() == ".docx":
                        def split_docx():
                            return [Document(page_content=t) for t in utils.docx_split(tmp_path)]

                        splits = await asyncio.to_thread(split_docx)
                    elif suffix.lower() == ".json":
                        def split_json():
                            with open(tmp_path, "r", encoding="utf-8") as f:
                                json_data = json.load(f)
                                return [Document(page_content=json.dumps(item)) for item in utils.json_split(json_data)]

                        splits = await asyncio.to_thread(split_json)
                    elif suffix.lower() in [".py", ".java", ".js", ".ts", ".vue", ".html", ".rb"]:
                        def split_code(lang):
                            with open(tmp_path, "r", encoding="utf-8") as f:
                                return [Document(page_content=t) for t in utils.code_split(f.read(), lang)]

                        lang_map = {
                            ".py": "python", ".java": "java", ".js": "js", ".ts": "js", ".vue": "js",
                            ".html": "html", ".rb": "ruby"
                        }
                        splits = await asyncio.to_thread(split_code, lang_map[suffix.lower()])
                    elif suffix.lower() in [".xml", ".yml", ".yaml", ".sh", ".css", ".scss"]:
                        def split_plain():
                            with open(tmp_path, "r", encoding="utf-8") as f:
                                return [Document(page_content=t) for t in utils.plain_text_split(f.read())]

                        splits = await asyncio.to_thread(split_plain)
                    elif suffix.lower() in IMAGE_MIME_TYPES:
                        # Process image directly from bytes
                        splits = await utils.image_split(
                            io.BytesIO(minio_byte),
                            mime_type=IMAGE_MIME_TYPES[suffix.lower()],
                        )
                    else:
                        raise ValueError(f"不支持的文件类型: {suffix or '未知'}")

                    if not splits:
                        raise ValueError("文档未解析出任何有效内容")

                    logger.info(f"Document {document_id} split into {len(splits)} chunks.")
                    for i, doc in enumerate(splits):
                        doc.metadata["documentId"] = document_id
                        doc.metadata["chunkIndex"] = i
                        doc.metadata["maxChunkIndex"] = len(splits) - 1
                        doc.metadata["fileName"] = file_name
                    # Embed and store
                    milvus_uri = os.environ.get("MILVUS_URI")
                    milvus_token = os.environ.get("MILVUS_TOKEN")
                    # Create vector store
                    embedding_config = {
                        'name': 'text-embedding-v4',
                        'provider': 'qwen'
                    }

                    embeddings = get_embedding_instance(embedding_config)
                    vector_store = await MilvusClientManager.get_instance(
                        user_id, kb_id, milvus_uri, milvus_token, embeddings
                    )
                    if vector_store is None:
                        raise RuntimeError("无法连接到知识库")

                    max_batch = 32
                    existing_ids = await asyncio.to_thread(
                        self._load_existing_vector_ids, vector_store, document_id
                    )
                    inserted_ids = []

                    def store_documents_batch():
                        for i in range(0, len(splits), max_batch):
                            inserted_ids.extend(
                                vector_store.add_documents(splits[i:i + max_batch])
                            )
                        return inserted_ids

                    async def rollback_inserted_ids():
                        if not inserted_ids:
                            return
                        try:
                            await asyncio.to_thread(
                                self._delete_vector_ids, vector_store, inserted_ids
                            )
                        except Exception as cleanup_error:
                            logger.error(f"回滚新向量失败: {cleanup_error}")

                    try:
                        ids = await asyncio.to_thread(store_documents_batch)
                    except Exception:
                        logger.error(f"Document {document_id} 写入向量失败，回滚本次已写入的切片")
                        await rollback_inserted_ids()
                        raise

                    if not ids or len(ids) != len(splits):
                        logger.error(
                            f"Document {document_id} 向量写入数量异常: "
                            f"{len(ids)}/{len(splits)}"
                        )
                        await rollback_inserted_ids()
                        raise RuntimeError("向量写入数量与切片数量不一致")

                    new_id_set = set()
                    for vector_id in ids:
                        try:
                            new_id_set.add(int(vector_id))
                        except (TypeError, ValueError):
                            continue
                    if ids and not new_id_set:
                        logger.warning(
                            f"Document {document_id} 无法解析新向量主键，跳过旧向量清理"
                        )
                        stale_ids = []
                    else:
                        stale_ids = [pk for pk in existing_ids if pk not in new_id_set]
                    if stale_ids:
                        logger.info(f"Document {document_id} 清理旧向量 {len(stale_ids)} 条")
                        await asyncio.to_thread(
                            self._delete_vector_ids, vector_store, stale_ids
                        )

                    logger.info(f"Document {document_id} processed and stored with {len(ids)} chunks.")
                    chunks_data = []
                    for i, (doc, vector_id) in enumerate(zip(splits, ids)):
                        chunks_data.append({
                            "chunkIndex": i,
                            "text": doc.page_content,
                            "tokenLength": len(doc.page_content),
                            "vectorId": str(vector_id),
                            "metadata": {}
                        })
                    # Send success message
                    response_message = {
                        "documentId": document_id,
                        "status": "success",
                        "message": "Document processed successfully",
                        "chunksCount": len(splits),
                        "chunks": chunks_data
                    }
                    await rabbit_async_client.publish(
                        exchange_name="server.interact.llm.exchange",
                        routing_key="rag.document.complete.key",
                        message=response_message
                    )
                except Exception as e:
                    logger.error(f"Error during embedding or storage: {e}")
                    error_stack = traceback.format_exc()
                    logger.error(error_stack)
                    await self.error_message_sender(document_id, str(e))
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                error_stack = traceback.format_exc()
                logger.error(error_stack)
                if document_id is not None:
                    await self.error_message_sender(document_id, str(e))
                else:
                    logger.error("消息缺少 documentId，无法回报失败状态，消息将被丢弃")


document_embedding_consumer = DocumentEmbeddingConsumer()
