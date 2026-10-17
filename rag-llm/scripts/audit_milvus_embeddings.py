#!/usr/bin/env python3
"""审计并迁移 Milvus collection 的 embedding 一致性。

背景：
    rag-llm 在 2026-03-09 从 DashScope text-embedding-v4 切换到本地
    Qwen/Qwen3-Embedding-0.6B。两者维度相同（1024）、COSINE，schema 完全兼容，
    但向量空间不同。旧 collection 如果未重新向量化，semantic_search 会近似随机。

默认行为（只读审计，不修改 Milvus）：
    对每个 kb_* collection 抽样若干 chunk，用当前 embedding 服务重算向量，
    与库内向量计算 cosine；输出 consistent / partial / stale / error 分类。
    对 stale/partial 的 collection，额外打印迁移计划（entities、批次数）。

迁移（默认仍是 dry-run）：
    --apply 只打印计划；必须同时加 --yes 才真正写入。
    Milvus 2.6.11 实测 auto_id collection 的 upsert 会新增主键并保留旧行
    （row_count +1），因此脚本对 auto_id collection 直接禁用 upsert；
    真实 collection 使用 --replace-mode delete-insert，并需要
    --allow-delete-insert。

用法：
    MILVUS_URI=http://... MILVUS_TOKEN=... \
        python scripts/audit_milvus_embeddings.py --dbs group_0 \
        --created-before 2026-03-09

    MILVUS_URI=... MILVUS_TOKEN=... \
        python scripts/audit_milvus_embeddings.py --dbs group_0 \
        --collections kb_178,kb_193 --apply --yes \
        --replace-mode delete-insert --allow-delete-insert

注意：
    - 脚本不打印 MILVUS_TOKEN。
    - 审计会按需 load collection，并在自己 load 的情况下 release；
      原本已 loaded 的 collection 不会被 release。
    - 迁移前会重新审计；只有 stale/partial 才会进入迁移。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from pymilvus import MilvusClient
from pymilvus.client.types import LoadState

DEFAULT_EMBEDDING_URL = "http://192.168.188.6:8890"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_THRESHOLD = 0.9
STATUSES = ("consistent", "partial", "stale", "empty", "skipped", "error")


def log(message: str) -> None:
    print(message, flush=True)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_created_before(value: str) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        try:
            parsed = dt.datetime.fromisoformat(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"无效日期 {value!r}，请使用 YYYY-MM-DD 或 ISO 格式"
            ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def collection_created_at(desc: dict[str, Any]) -> Optional[dt.datetime]:
    value = desc.get("collection_id")
    if value is None:
        return None
    try:
        # Milvus collection_id 是 hybrid timestamp：高 18 位后是毫秒时间戳
        return dt.datetime.fromtimestamp(
            (int(value) >> 18) / 1000, tz=dt.timezone.utc
        )
    except (TypeError, ValueError, OSError):
        return None


def vector_field_name(desc: dict[str, Any]) -> str:
    fields = desc.get("fields") or []
    for field in fields:
        if field.get("name") == "vector":
            return "vector"
    for field in fields:
        field_type = str(field.get("type", ""))
        if "FLOAT_VECTOR" in field_type or field_type == "101":
            return field["name"]
    raise RuntimeError("collection 中没有找到 FLOAT_VECTOR 字段")


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError(f"向量维度不一致: {len(left)} vs {len(right)}")
    dot = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(a * a for a in left))
    right_norm = math.sqrt(math.fsum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def embed_texts(
        texts: list[str],
        url: str,
        model: str,
        timeout: float,
) -> list[list[float]]:
    if not texts:
        return []
    payload = json.dumps(
        {"input": texts, "model": model},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"embedding 服务返回 HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"embedding 服务连接失败: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("embedding 服务返回了非 JSON 响应") from exc

    data = body.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        received = len(data) if isinstance(data, list) else "invalid"
        raise RuntimeError(f"embedding 返回数量不一致: {received}/{len(texts)}")
    data = sorted(data, key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in data]


def embed_in_batches(
        texts: list[str],
        url: str,
        model: str,
        timeout: float,
        batch_size: int,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        embeddings.extend(
            embed_texts(texts[start:start + batch_size], url, model, timeout)
        )
    return embeddings


def load_state(client: MilvusClient, collection: str) -> Any:
    try:
        response = client.get_load_state(collection)
    except Exception as exc:
        raise RuntimeError(f"读取 load state 失败: {exc}") from exc
    return response.get("state") if isinstance(response, dict) else None


def is_loaded(state: Any) -> bool:
    return state == LoadState.Loaded or str(state).endswith("Loaded")


def is_loading(state: Any) -> bool:
    return state == LoadState.Loading or str(state).endswith("Loading")


def wait_loaded(client: MilvusClient, collection: str, timeout: float) -> None:
    deadline = time.time() + max(timeout, 0)
    while True:
        if is_loaded(load_state(client, collection)):
            return
        if time.time() >= deadline:
            raise RuntimeError(f"collection load 超时（{timeout:g}s）")
        time.sleep(1.0)


def ensure_loaded(client: MilvusClient, collection: str, timeout: float) -> bool:
    """确保 collection 已加载；返回 True 表示是脚本自己 load 的。"""
    state = load_state(client, collection)
    if is_loaded(state):
        return False
    if is_loading(state):
        wait_loaded(client, collection, timeout)
        return False
    try:
        client.load_collection(collection)
    except Exception as exc:
        if "already loaded" not in str(exc).lower():
            raise
    wait_loaded(client, collection, timeout)
    return True


def release_if_needed(
        client: MilvusClient,
        collection: str,
        loaded_by_us: bool,
) -> None:
    if not loaded_by_us:
        return
    try:
        client.release_collection(collection)
    except Exception as exc:
        log(f"  [WARN] release {collection} 失败: {exc}")


def classify(scores: list[float], threshold: float) -> str:
    if not scores:
        return "error"
    if min(scores) >= threshold:
        return "consistent"
    if max(scores) < threshold:
        return "stale"
    return "partial"


def logical_count(client: MilvusClient, collection: str) -> int:
    """按 pk >= 0 统计逻辑行数；不用 get_collection_stats（delete 后含 tombstone）。"""
    rows = client.query(
        collection_name=collection,
        filter="pk >= 0",
        output_fields=["count(*)"],
    )
    if not rows:
        return 0
    return int(rows[0].get("count(*)", rows[0].get("count", 0)))


def audit_collection(
        client: MilvusClient,
        db: str,
        collection: str,
        desc: dict[str, Any],
        args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "db": db,
        "collection": collection,
        "created_at": None,
        "entities": None,
        "scores": [],
        "status": "error",
        "error": None,
    }
    created = collection_created_at(desc)
    if created is not None:
        result["created_at"] = created.strftime("%Y-%m-%d %H:%M:%S")

    try:
        stats = client.get_collection_stats(collection)
        result["entities"] = int(stats.get("row_count", 0))
    except Exception as exc:
        result["error"] = f"读取 row_count 失败: {exc}"
        return result

    if result["entities"] == 0:
        result["status"] = "empty"
        return result

    loaded_by_us = False
    try:
        state = load_state(client, collection)
        if args.only_loaded and not is_loaded(state):
            result["status"] = "skipped"
            return result

        loaded_by_us = ensure_loaded(client, collection, args.load_timeout)
        try:
            # get_collection_stats 在 delete 后会包含 tombstone；
            # collection 已加载时用 count(*) 得到逻辑行数。
            result["entities"] = logical_count(client, collection)
        except Exception:
            pass
        vector_field = vector_field_name(desc)
        rows = client.query(
            collection_name=collection,
            filter="pk >= 0",
            output_fields=["pk", "text", vector_field],
            limit=args.sample_size,
        )

        samples: list[tuple[str, list[float]]] = []
        for row in rows:
            text = row.get("text") or ""
            vector = row.get(vector_field) or []
            if not text.strip() or not vector:
                continue
            samples.append((text, vector))
        if not samples:
            result["error"] = "没有可用的非空样本"
            return result

        new_vectors = embed_in_batches(
            [text for text, _ in samples],
            args.embedding_url,
            args.embedding_model,
            args.embed_timeout,
            args.embed_batch_size,
        )
        scores = [
            cosine(stored, new)
            for (_, stored), new in zip(samples, new_vectors)
        ]
        result["scores"] = [round(score, 6) for score in scores]
        result["status"] = classify(scores, args.threshold)
    except Exception as exc:
        result["error"] = str(exc)
        result["status"] = "error"
    finally:
        release_if_needed(client, collection, loaded_by_us)
    return result


def migrate_collection(
        client: MilvusClient,
        db: str,
        collection: str,
        desc: dict[str, Any],
        audit: dict[str, Any],
        args: argparse.Namespace,
) -> dict[str, Any]:
    outcome = {
        "db": db,
        "collection": collection,
        "migrated": False,
        "verified": False,
        "error": None,
    }
    if audit["status"] not in ("stale", "partial"):
        return outcome

    entities = audit.get("entities") or 0
    batches = math.ceil(entities / args.batch_size) if entities else 0
    plan = (
        f"[PLAN] {db}.{collection} status={audit['status']} "
        f"entities={entities} batches={batches}"
    )
    if not args.apply:
        log(plan)
        return outcome
    if not args.yes:
        log(plan + "（未写入；加 --yes 执行）")
        return outcome

    auto_id = bool(desc.get("auto_id"))
    if args.replace_mode == "upsert" and auto_id:
        log(
            f"[SKIP] {db}.{collection}: Milvus 2.6.11 实测 auto_id upsert "
            f"会新增主键并保留旧行；请改用 --replace-mode delete-insert"
        )
        return outcome
    if args.replace_mode == "delete-insert" and not args.allow_delete_insert:
        log(
            f"[SKIP] {db}.{collection}: delete-insert 会短暂产生重复数据；"
            f"确认试点后加 --allow-delete-insert"
        )
        return outcome

    loaded_by_us = False
    try:
        loaded_by_us = ensure_loaded(client, collection, args.load_timeout)
        vector_field = vector_field_name(desc)
        payload_fields = [
            field["name"]
            for field in desc.get("fields", [])
            if field["name"] not in ("pk", vector_field)
        ]
        if "text" not in payload_fields:
            raise RuntimeError("collection 缺少 text 字段，无法重嵌入")

        try:
            before_entities = logical_count(client, collection)
        except Exception as exc:
            before_entities = audit.get("entities")
            log(f"  [WARN] {db}.{collection}: 读取迁移前逻辑行数失败: {exc}")
        iterator = client.query_iterator(
            collection_name=collection,
            batch_size=args.batch_size,
            filter="pk >= 0",
            output_fields=["pk", *payload_fields],
        )
        migrated_rows = 0
        skipped_rows = 0
        batch_no = 0
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                batch_no += 1
                texts = [row.get("text") or "" for row in batch]
                valid_indices = [
                    index for index, text in enumerate(texts) if text.strip()
                ]
                skipped_rows += len(batch) - len(valid_indices)
                if not valid_indices:
                    continue
                new_vectors = embed_in_batches(
                    [texts[index] for index in valid_indices],
                    args.embedding_url,
                    args.embedding_model,
                    args.embed_timeout,
                    args.embed_batch_size,
                )
                insert_data = []
                for index, vector in zip(valid_indices, new_vectors):
                    row = batch[index]
                    item = {field: row.get(field) for field in payload_fields}
                    item[vector_field] = vector
                    insert_data.append(item)

                if args.replace_mode == "upsert":
                    upsert_data = [
                        {**item, "pk": batch[index]["pk"]}
                        for item, index in zip(insert_data, valid_indices)
                    ]
                    response = client.upsert(
                        collection_name=collection, data=upsert_data
                    )
                    written = int(response.get("upsert_count", len(upsert_data)))
                    if written != len(upsert_data):
                        raise RuntimeError(
                            f"upsert 数量不一致: {written}/{len(upsert_data)}"
                        )
                else:
                    # 先插入新行，再删除旧行：插入失败时旧数据仍在；
                    # 删除失败时会短期重复，脚本会报错停止。
                    response = client.insert(
                        collection_name=collection, data=insert_data
                    )
                    new_ids = response.get("ids") or []
                    if len(new_ids) != len(insert_data):
                        raise RuntimeError(
                            f"insert 数量不一致: {len(new_ids)}/{len(insert_data)}"
                        )
                    old_ids = [batch[index]["pk"] for index in valid_indices]
                    client.delete(collection_name=collection, ids=old_ids)

                migrated_rows += len(insert_data)
                log(
                    f"  [MIGRATE] {db}.{collection} batch={batch_no} "
                    f"rows={len(insert_data)} total={migrated_rows}"
                )
        finally:
            iterator.close()

        outcome["migrated"] = True
        if skipped_rows:
            log(f"  [WARN] {db}.{collection}: 跳过 {skipped_rows} 个空 text 切片")

        # delete-insert 的删除在 Milvus 中需要 flush/短暂等待才对查询可见；
        # 否则复检可能仍抽到旧行，误判为 stale。
        try:
            client.flush(collection_name=collection)
        except Exception as exc:
            log(f"  [WARN] {db}.{collection}: 迁移后 flush 失败: {exc}")
        if args.verify_wait > 0:
            time.sleep(args.verify_wait)

        # 迁移后复检：抽样重算，确认 self-cosine 已恢复。
        verify = audit_collection(client, db, collection, desc, args)
        if verify["status"] == "consistent":
            outcome["verified"] = True
            log(
                f"  [VERIFY_OK] {db}.{collection} "
                f"scores={verify['scores']}"
            )
        else:
            outcome["error"] = (
                f"迁移后复检未通过: status={verify['status']} "
                f"scores={verify['scores']} error={verify['error']}"
            )
            log(f"  [VERIFY_FAIL] {db}.{collection}: {outcome['error']}")

        if before_entities is not None:
            try:
                after_entities = logical_count(client, collection)
                if after_entities != before_entities:
                    log(
                        f"  [WARN] {db}.{collection}: 逻辑行数 "
                        f"{before_entities} -> {after_entities}"
                    )
            except Exception as exc:
                log(f"  [WARN] {db}.{collection}: 读取迁移后逻辑行数失败: {exc}")
    except Exception as exc:
        outcome["error"] = str(exc)
        log(f"  [ERROR] {db}.{collection} 迁移失败: {exc}")
    finally:
        release_if_needed(client, collection, loaded_by_us)
    return outcome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="审计 Milvus collection 的 embedding 一致性；默认只读，不修改数据"
    )
    parser.add_argument("--dbs", default="", help="逗号分隔的数据库名；默认所有 group_*")
    parser.add_argument("--collections", default="", help="逗号分隔的 collection 名")
    parser.add_argument(
        "--created-before",
        default="",
        help="只处理 collection_id 早于此日期的 collection，格式 YYYY-MM-DD",
    )
    parser.add_argument("--sample-size", type=int, default=3, help="每个 collection 抽样数量")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"self-cosine 判定阈值，默认 {DEFAULT_THRESHOLD}",
    )
    parser.add_argument(
        "--embedding-url",
        default=os.environ.get("EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_URL),
        help="当前 embedding 服务地址",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="embedding 模型名（服务端当前忽略该字段）",
    )
    parser.add_argument("--embed-batch-size", type=int, default=32, help="迁移时 embedding 批大小")
    parser.add_argument("--embed-timeout", type=float, default=60, help="embedding 请求超时秒数")
    parser.add_argument("--batch-size", type=int, default=32, help="迁移时 Milvus 查询/写入批大小")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="进入迁移模式；默认只审计并打印迁移计划",
    )
    parser.add_argument("--yes", action="store_true", help="确认真正写入")
    parser.add_argument(
        "--replace-mode",
        choices=("upsert", "delete-insert"),
        default="upsert",
        help="向量替换方式；默认 upsert",
    )
    parser.add_argument(
        "--allow-delete-insert",
        action="store_true",
        help="允许 delete-insert 模式（会短暂产生重复行）",
    )
    parser.add_argument(
        "--only-loaded",
        action="store_true",
        help="只审计当前已加载的 collection，不主动 load",
    )
    parser.add_argument("--load-timeout", type=float, default=60, help="collection load 超时秒数")
    parser.add_argument(
        "--verify-wait",
        type=float,
        default=3.0,
        help="迁移后等待 delete 可见的秒数，默认 3",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="每个 collection 之间的等待秒数")
    parser.add_argument(
        "--max-collections",
        type=int,
        default=0,
        help="最多处理多少个 collection；0 表示不限制",
    )
    parser.add_argument("--json-out", default="", help="把审计结果写入 JSON 文件")
    parser.add_argument("--verbose", action="store_true", help="打印 consistent/empty 等所有结果")
    parser.add_argument(
        "--fail-on-stale",
        action="store_true",
        help="发现 stale/partial 时以退出码 2 结束",
    )
    args = parser.parse_args()

    if args.sample_size < 1:
        parser.error("--sample-size 必须 >= 1")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold 必须在 [0, 1] 之间")
    if args.embed_batch_size < 1 or args.batch_size < 1:
        parser.error("--embed-batch-size/--batch-size 必须 >= 1")
    if args.embed_timeout <= 0 or args.load_timeout <= 0:
        parser.error("--embed-timeout/--load-timeout 必须 > 0")
    if args.verify_wait < 0:
        parser.error("--verify-wait 必须 >= 0")
    try:
        args.created_before = parse_created_before(args.created_before)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    args = parse_args()
    uri = os.environ.get("MILVUS_URI")
    token = os.environ.get("MILVUS_TOKEN")
    if not uri:
        print("缺少 MILVUS_URI 环境变量", file=sys.stderr)
        return 2

    root = MilvusClient(uri=uri, token=token)
    try:
        databases = [
            item["name"] if isinstance(item, dict) else item
            for item in root.list_databases()
        ]
    finally:
        root.close()

    dbs = split_csv(args.dbs)
    if not dbs:
        dbs = sorted(name for name in databases if name.startswith("group_"))
    if not dbs:
        print("没有发现 group_* 数据库", file=sys.stderr)
        return 1

    only_collections = set(split_csv(args.collections))
    results: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    counts = {status: 0 for status in STATUSES}
    checked = 0
    planned_batches = 0
    stale_entities = 0
    partial_entities = 0
    errors = 0

    for db in dbs:
        client = MilvusClient(uri=uri, token=token, db_name=db)
        try:
            collections = sorted(
                name
                for name in client.list_collections()
                if name.startswith("kb_")
                and (not only_collections or name in only_collections)
            )
            log(f"DB {db}: {len(collections)} 个候选 collection")
            for collection in collections:
                if args.max_collections and checked >= args.max_collections:
                    break
                checked += 1
                try:
                    desc = client.describe_collection(collection)
                except Exception as exc:
                    errors += 1
                    counts["error"] += 1
                    result = {
                        "db": db,
                        "collection": collection,
                        "created_at": None,
                        "entities": None,
                        "scores": [],
                        "status": "error",
                        "error": f"describe_collection 失败: {exc}",
                    }
                    results.append(result)
                    log(f"[ERROR] {db}.{collection}: {result['error']}")
                    continue

                created = collection_created_at(desc)
                if args.created_before is not None:
                    if created is None:
                        log(
                            f"[WARN] {db}.{collection}: collection_id 缺失，"
                            f"无法应用 --created-before，仍纳入审计"
                        )
                    elif created >= args.created_before:
                        counts["skipped"] += 1
                        continue

                audit = audit_collection(client, db, collection, desc, args)
                results.append(audit)
                status = audit["status"]
                counts[status] = counts.get(status, 0) + 1
                if status == "stale":
                    stale_entities += audit.get("entities") or 0
                elif status == "partial":
                    partial_entities += audit.get("entities") or 0
                if status == "error":
                    errors += 1

                if status in ("stale", "partial", "error") or args.verbose:
                    created_text = audit.get("created_at") or "unknown"
                    log(
                        f"[{status.upper()}] {db}.{collection} "
                        f"created={created_text} entities={audit.get('entities')} "
                        f"scores={audit.get('scores')}"
                        + (f" error={audit.get('error')}" if audit.get("error") else "")
                    )

                if status in ("stale", "partial"):
                    entities = audit.get("entities") or 0
                    planned_batches += math.ceil(entities / args.batch_size) if entities else 0
                    outcome = migrate_collection(client, db, collection, desc, audit, args)
                    outcomes.append(outcome)
                    if outcome.get("error"):
                        errors += 1
                time.sleep(max(args.sleep, 0))
        finally:
            client.close()

    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_url": args.embedding_url,
        "embedding_model": args.embedding_model,
        "threshold": args.threshold,
        "created_before": (
            args.created_before.strftime("%Y-%m-%d") if args.created_before else None
        ),
        "checked": checked,
        "counts": counts,
        "stale_entities": stale_entities,
        "partial_entities": partial_entities,
        "planned_batches": planned_batches,
        "apply": bool(args.apply and args.yes),
        "errors": errors,
    }
    if args.json_out:
        payload = {"summary": summary, "collections": results, "migrations": outcomes}
        try:
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            log(f"审计结果已写入 {args.json_out}")
        except OSError as exc:
            errors += 1
            log(f"[ERROR] 写入 {args.json_out} 失败: {exc}")

    log(
        "完成：检查 {checked}，一致 {consistent}，部分 {partial}，过期 {stale}，"
        "空 {empty}，跳过 {skipped}，错误 {error}；"
        "过期/部分 chunk 数 {chunks}，预计迁移批次 {batches}{mode}".format(
            checked=checked,
            chunks=stale_entities + partial_entities,
            batches=planned_batches,
            mode="" if args.apply and args.yes else "（dry-run，未修改 Milvus）",
            **counts,
        )
    )
    if args.fail_on_stale and (counts["stale"] or counts["partial"]):
        return 2
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
