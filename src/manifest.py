"""ingest manifest：记录每个已入库文件的状态，支撑增量 ingest。

变更检测信号 = mtime + size（用户要的"时间戳"，加 size 防止 mtime 相同但内容变）。
不做内容哈希：那要读全文件，而解析本就是瓶颈，mtime+size 足够且零额外开销。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "ingest_manifest.json")


def _file_state(path: str) -> dict:
    st = os.stat(path)
    return {"mtime": st.st_mtime, "size": st.st_size}


def load_manifest(path: str = _MANIFEST_PATH) -> dict:
    """返回 {filename: {"mtime", "size"}}；不存在则空 dict（首次/老库自然当全新增）。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(manifest: dict, path: str = _MANIFEST_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


@dataclass
class Diff:
    added: list[str] = field(default_factory=list)      # 磁盘有、manifest 无
    modified: list[str] = field(default_factory=list)   # mtime/size 变了
    deleted: list[str] = field(default_factory=list)    # manifest 有、磁盘无
    unchanged: list[str] = field(default_factory=list)  # 完全一致

    @property
    def to_parse(self) -> list[str]:
        """需要重新解析的文件（新增 + 修改）。"""
        return self.added + self.modified

    @property
    def to_delete_points(self) -> list[str]:
        """需要先删旧点的文件（修改 + 删除）。"""
        return self.modified + self.deleted


def diff_data_dir(data_dir: str, manifest: dict) -> tuple[Diff, dict]:
    """对比磁盘当前状态与 manifest，返回 (Diff, 新 manifest)。

    新 manifest 只包含当前磁盘上存在的文件，可直接 save。
    """
    current_files = sorted(
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
    )
    diff = Diff()
    new_manifest = {}

    for fname in current_files:
        state = _file_state(os.path.join(data_dir, fname))
        new_manifest[fname] = state
        old = manifest.get(fname)
        if old is None:
            diff.added.append(fname)
        elif old.get("mtime") != state["mtime"] or old.get("size") != state["size"]:
            diff.modified.append(fname)
        else:
            diff.unchanged.append(fname)

    # manifest 有但磁盘没了 → 删除
    for fname in manifest:
        if fname not in new_manifest:
            diff.deleted.append(fname)

    return diff, new_manifest
