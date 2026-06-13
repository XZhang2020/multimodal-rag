"""content-hash 磁盘缓存。

OCR 和 VLM 都是慢且（VLM）花钱的操作。同一篇文档反复 ingest 时，
按"图片/页面字节内容的哈希"缓存结果，命中就跳过，避免重复付费。

缓存键 = sha1(payload_bytes + namespace)，与文件名/页码无关，
所以同一张图在不同文档里也能命中。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Optional

_DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".parse_cache")
_LOCK = threading.Lock()


class ParseCache:
    def __init__(self, cache_dir: str = _DEFAULT_DIR, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = os.path.abspath(cache_dir)
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _key(self, payload: bytes, namespace: str) -> str:
        h = hashlib.sha1()
        h.update(namespace.encode("utf-8"))
        h.update(b"::")
        h.update(payload)
        return h.hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, payload: bytes, namespace: str) -> Optional[str]:
        if not self.enabled:
            return None
        path = self._path(self._key(payload, namespace))
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("value")
        except Exception:
            return None

    def set(self, payload: bytes, namespace: str, value: str) -> None:
        if not self.enabled:
            return
        path = self._path(self._key(payload, namespace))
        try:
            with _LOCK:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"value": value, "ts": time.time()}, f, ensure_ascii=False)
        except Exception:
            # 缓存写失败绝不影响主流程
            pass

    def is_fresh(self, payload: bytes, namespace: str, ttl: float) -> bool:
        """该 key 是否在 ttl 秒内被 set 过。用于"失败标记"等临时缓存：
        临时故障（如本地 VLM 偶发 OOM）TTL 内跳过重试，过期后自动允许重试。"""
        if not self.enabled or ttl <= 0:
            return False
        path = self._path(self._key(payload, namespace))
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                ts = json.load(f).get("ts", 0)
            return (time.time() - ts) < ttl
        except Exception:
            return False
