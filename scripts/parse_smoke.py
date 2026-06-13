"""解析层冒烟测试：打印各模态元素分布，肉眼验证表格/图片/扫描页是否被正确处理。

用法：
    conda activate handson_llm
    python scripts/parse_smoke.py            # 解析 data/ 下所有文件
    python scripts/parse_smoke.py x.pdf      # 解析指定文件
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsing import parse_elements


def show(path):
    print("=" * 70)
    print(f"文件: {os.path.basename(path)}")
    print("=" * 70)
    elements = parse_elements(path)
    by_mod = Counter(e.modality for e in elements)
    print(f"元素总数: {len(elements)} | 分布: {dict(by_mod)}")

    # 每种模态各抽一个样本看效果
    seen = set()
    for e in elements:
        if e.modality in seen:
            continue
        seen.add(e.modality)
        preview = (e.text or "")[:160].replace("\n", " ")
        print(f"\n[{e.modality}] p.{e.page} id={e.element_id}")
        print(f"  检索文本: {preview}")
        if e.modality == "table" and e.raw:
            print(f"  markdown 原表(前120字): {e.raw[:120]}")
    print()


def main():
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        targets = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))
                   if os.path.isfile(os.path.join(data_dir, f))]
    for t in targets:
        try:
            show(t)
        except Exception as e:
            import traceback
            print(f"[FAIL] {t}: {e!r}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
