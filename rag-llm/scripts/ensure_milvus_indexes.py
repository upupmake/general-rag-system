#!/usr/bin/env python3
"""检查并补齐知识库 collection 的标量索引。

默认只做 dry-run，不修改 Milvus；确认输出后加 --apply 才会创建索引。

用法：
    MILVUS_URI=http://... MILVUS_TOKEN=... python scripts/ensure_milvus_indexes.py
    MILVUS_URI=http://... MILVUS_TOKEN=... python scripts/ensure_milvus_indexes.py --apply

注意：
- 老 collection 可能缺少 text NGRAM / fileName INVERTED / documentId STL_SORT，
  会导致关键词检索退化为全表扫描、find_files 的 order_by 变慢。
- 若 collection 处于 loaded 状态导致建索引失败，脚本默认只报告；确认可以短暂释放后，
  加 --release-loaded 让脚本 release -> create_index，后续由服务按需重新 load。
"""
import argparse
import os
import sys
import time

from pymilvus import MilvusClient

REQUIRED_INDEXES = {
    "documentId": {"index_type": "STL_SORT"},
    "fileName": {"index_type": "INVERTED"},
    # 与现有 collection 保持一致：中文 2~3 gram 的 NGRAM 索引
    "text": {"index_type": "NGRAM", "min_gram": 2, "max_gram": 3},
}


def parse_args():
    parser = argparse.ArgumentParser(description="检查并补齐 Milvus 知识库 collection 的标量索引")
    parser.add_argument("--apply", action="store_true", help="真正创建缺失索引；默认只 dry-run")
    parser.add_argument("--dbs", default="", help="逗号分隔的数据库名；默认自动发现 group_*")
    parser.add_argument("--collections", default="", help="逗号分隔的 collection 名；默认处理所有 kb_*")
    parser.add_argument("--sleep", type=float, default=0.2, help="每个 collection 之间的等待秒数")
    parser.add_argument(
        "--release-loaded",
        action="store_true",
        help="建索引遇到 loaded collection 时先 release 再重试；会短暂影响该 collection 的查询",
    )
    return parser.parse_args()


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _create_index(client: MilvusClient, collection: str, field_name: str, params: dict) -> None:
    index_params = client.prepare_index_params()
    index_params.add_index(field_name=field_name, **params)
    client.create_index(collection_name=collection, index_params=index_params)


def main() -> int:
    args = parse_args()
    uri = os.environ.get("MILVUS_URI")
    token = os.environ.get("MILVUS_TOKEN")
    if not uri:
        print("缺少 MILVUS_URI 环境变量", file=sys.stderr)
        return 2

    root = MilvusClient(uri=uri, token=token)
    dbs = _split(args.dbs)
    if not dbs:
        dbs = sorted(name for name in root.list_databases() if name.startswith("group_"))
    if not dbs:
        print("没有发现 group_* 数据库", file=sys.stderr)
        return 1

    only_collections = set(_split(args.collections))
    checked = 0
    missing_count = 0
    created_count = 0
    failed_count = 0

    for db in dbs:
        client = MilvusClient(uri=uri, token=token, db_name=db)
        collections = [
            name for name in client.list_collections()
            if name.startswith("kb_") and (not only_collections or name in only_collections)
        ]
        print(f"DB {db}: {len(collections)} 个 collection")
        for collection in collections:
            checked += 1
            try:
                indexes = set(client.list_indexes(collection))
                fields = {
                    item["name"]
                    for item in client.describe_collection(collection).get("fields", [])
                }
            except Exception as exc:
                failed_count += 1
                print(f"  [ERROR] {db}.{collection}: 读取 schema/index 失败: {exc}")
                continue

            missing = [
                (field_name, params)
                for field_name, params in REQUIRED_INDEXES.items()
                if field_name in fields and field_name not in indexes
            ]
            if not missing:
                continue

            missing_count += 1
            description = ", ".join(
                f"{field_name}({params['index_type']})"
                for field_name, params in missing
            )
            if not args.apply:
                print(f"  [DRY-RUN] {db}.{collection}: 缺少 {description}")
                continue

            print(f"  [APPLY] {db}.{collection}: 创建 {description}")
            for field_name, params in missing:
                try:
                    _create_index(client, collection, field_name, params)
                    created_count += 1
                except Exception as exc:
                    message = str(exc)
                    if args.release_loaded and "loaded" in message.lower():
                        try:
                            print(f"    [RELEASE] {db}.{collection} 后重试 {field_name}")
                            client.release_collection(collection)
                            _create_index(client, collection, field_name, params)
                            created_count += 1
                            continue
                        except Exception as retry_exc:
                            message = str(retry_exc)
                    failed_count += 1
                    print(f"    [ERROR] {db}.{collection}.{field_name}: {message}")
            time.sleep(max(args.sleep, 0))

    print(
        f"完成：检查 {checked} 个 collection，缺失 {missing_count} 个，"
        f"创建 {created_count} 个索引，失败 {failed_count} 个"
        + ("" if args.apply else "（dry-run，未修改 Milvus）")
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
