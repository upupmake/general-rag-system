#!/usr/bin/env python3
"""把本地文件上传到知识库（multipart 直传，文件字节不经过模型上下文）。

只需要 Python 3.9+ 标准库。

用法：
  python upload_kb_file.py --list
  python upload_kb_file.py --kb-id 12 --file "D:/docs/手册.pdf"
  python upload_kb_file.py --kb-id 12 --file a.pdf --file b.docx
  python upload_kb_file.py --kb-id 12 --file a.pdf --name "docs/规范/手册.pdf"
  python upload_kb_file.py --kb-id 12 --file a.pdf --dry-run

环境变量：
  RAG_ACCESS_KEY    必填，grs_ak_ 开头的 Access Key（也可用 --access-key）
  RAG_OPENAPI_BASE  可选，默认 https://starvpn.forwardforever.top:5616/api/openapi/v1
"""

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_BASE = "https://starvpn.forwardforever.top:5616/api/openapi/v1"


class _MultipartBody:
    """流式 multipart 请求体，避免把整个文件读进内存。"""

    def __init__(self, boundary: str, parts: list[tuple[str, Path]]) -> None:
        self.boundary = boundary
        self._parts: list[tuple[bytes, Path, int]] = []
        for name, path in parts:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            header = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="files"; filename="{name}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
            self._parts.append((header, path, path.stat().st_size))
        self._tail = f"--{boundary}--\r\n".encode("utf-8")
        self.size = sum(len(header) + size + 2 for header, _, size in self._parts) + len(self._tail)
        self._segments_iter = None
        self._buffer = b""

    def __len__(self) -> int:
        return self.size

    def _segments(self):
        for header, path, _size in self._parts:
            yield header
            with path.open("rb") as handle:
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        break
                    yield block
            yield b"\r\n"
        yield self._tail

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size
        if self._segments_iter is None:
            self._segments_iter = self._segments()
        while len(self._buffer) < size:
            try:
                self._buffer += next(self._segments_iter)
            except StopIteration:
                break
        data = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return data


def _base_url(override: str | None) -> str:
    return (override or os.getenv("RAG_OPENAPI_BASE") or DEFAULT_BASE).rstrip("/")


def _access_key(override: str | None) -> str:
    key = (override or os.getenv("RAG_ACCESS_KEY") or "").strip()
    if not key:
        sys.exit("缺少 Access Key：设置环境变量 RAG_ACCESS_KEY，或传 --access-key")
    return key


def _request(method: str, url: str, access_key: str, body=None, timeout: float = 30.0):
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("Authorization", f"Bearer {access_key}")
    request.add_header("Accept", "application/json")
    if body is not None:
        request.add_header("Content-Type", f"multipart/form-data; boundary={body.boundary}")
        request.add_header("Content-Length", str(body.size))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        sys.exit(f"连不上 {url}：{exc.reason}")


def _unwrap(status: int, payload: bytes):
    try:
        envelope = json.loads(payload.decode("utf-8"))
    except ValueError:
        sys.exit(f"服务返回非 JSON 响应（HTTP {status}）：{payload[:200]!r}")
    if envelope.get("code") != 200:
        sys.exit(f"请求失败（HTTP {status}）：{envelope.get('message') or envelope}")
    return envelope.get("data")


def _list_knowledge_bases(url_base: str, access_key: str) -> None:
    status, payload = _request("GET", f"{url_base}/knowledge-bases", access_key)
    data = _unwrap(status, payload) or {}
    groups = [
        ("owned（本人创建）", data.get("owned") or []),
        ("workspaceShared（工作空间共享）", data.get("workspaceShared") or []),
        ("invited（受邀）", data.get("invited") or []),
        ("public（公开）", data.get("public") or []),
    ]
    for title, items in groups:
        print(f"\n{title}:")
        if not items:
            print("  （空）")
            continue
        for item in items:
            kb_id = item.get("id")
            name = item.get("name")
            visibility = item.get("visibility")
            uploadable = title.startswith("owned") and visibility == "private"
            mark = "  [可上传]" if uploadable else ""
            print(f"  id={kb_id}  name={name}  visibility={visibility}{mark}")
    print("\n只有「本人创建 + visibility=private」的库能通过这个接口上传文件。")


def _upload(url_base: str, access_key: str, kb_id: int, parts: list[tuple[str, Path]],
            dry_run: bool) -> None:
    body = _MultipartBody(uuid.uuid4().hex, parts)
    url = f"{url_base}/knowledge-bases/{kb_id}/documents"
    if dry_run:
        print(json.dumps({
            "url": url,
            "method": "POST",
            "contentLength": body.size,
            "boundary": body.boundary,
            "files": [{"name": name, "path": str(path), "size": path.stat().st_size}
                      for name, path in parts],
        }, ensure_ascii=False, indent=2))
        return
    status, payload = _request("POST", url, access_key, body=body, timeout=300.0)
    documents = _unwrap(status, payload)
    print(json.dumps({
        "knowledgeBaseId": kb_id,
        "documents": documents,
        "note": "文档还需要后台解析/切片/向量化，刚上传时 status 可能不是就绪态；检索前可先用 find_knowledge_base_files 确认。",
    }, ensure_ascii=False, indent=2))


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="把本地文件上传到知识库（仅限本人创建的个人私有知识库）。")
    parser.add_argument("--list", action="store_true", dest="list_kbs",
                        help="列出当前账号可访问的知识库，标出可上传的库")
    parser.add_argument("--kb-id", type=int, help="目标知识库 ID")
    parser.add_argument("--file", action="append", default=[], dest="files",
                        help="要上传的本地文件路径，可重复传入")
    parser.add_argument("--name", help="远端文件名/相对路径（只允许单个 --file），默认用本地文件名")
    parser.add_argument("--access-key", help="Access Key；不传则读环境变量 RAG_ACCESS_KEY")
    parser.add_argument("--base-url", help="OpenAPI 基础地址；不传则读 RAG_OPENAPI_BASE")
    parser.add_argument("--dry-run", action="store_true", help="只组装请求并打印大小，不发网络请求")
    args = parser.parse_args()

    url_base = _base_url(args.base_url)

    if args.list_kbs:
        _list_knowledge_bases(url_base, _access_key(args.access_key))
        return

    if not args.kb_id:
        parser.error("上传需要 --kb-id（可先用 --list 查看）")
    if not args.files:
        parser.error("上传需要至少一个 --file")
    if args.name and len(args.files) != 1:
        parser.error("--name 只能和单个 --file 搭配使用")

    parts: list[tuple[str, Path]] = []
    for index, raw_path in enumerate(args.files):
        path = Path(raw_path).expanduser()
        if not path.is_file():
            sys.exit(f"文件不存在或不是普通文件：{path}")
        remote_name = args.name if args.name else path.name
        remote_name = remote_name.replace("\\", "/").replace('"', "'").replace("\r", " ").replace("\n", " ")
        parts.append((remote_name, path))

    access_key = "" if args.dry_run else _access_key(args.access_key)
    _upload(url_base, access_key, args.kb_id, parts, args.dry_run)


if __name__ == "__main__":
    main()
