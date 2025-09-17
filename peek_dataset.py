# peek_dataset.py
import os
import json
import argparse
from typing import List, Optional

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

TEXT_CANDIDATES = ["text", "content", "prompt", "input", "instruction"]


def try_load_dataset(data: str, split: Optional[str]):
    """Load HF repo id or local load_from_disk path."""
    if os.path.exists(data):  # local dataset dir (load_from_disk)
        ds = load_from_disk(data)
    else:  # HF hub repo id
        # 如需要 streaming，可加 streaming=True（但不支持 select/map 全功能）
        ds = load_dataset(data)

    if isinstance(ds, DatasetDict):
        if split is None:
            # 默认优先 train，退而求其次第一个可用的 split
            split = "train" if "train" in ds else list(ds.keys())[0]
        return ds[split], ds
    elif isinstance(ds, Dataset):
        return ds, ds
    else:
        raise ValueError(f"Unsupported dataset type: {type(ds)}")


def shorten(x, max_len=160):
    s = json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
    return (s[:max_len] + " ...") if len(s) > max_len else s


def guess_text_column(cols: List[str]):
    for c in TEXT_CANDIDATES:
        if c in cols:
            return c
    return None


def print_basic_info(ds: Dataset, full):
    print("=== Basic Info ===")
    try:
        print("Num rows:", len(ds))
    except Exception:
        print("Num rows: <unknown> (streaming?)")
    print("Columns:", ds.column_names)
    print()

    # 取样例
    print("=== First Rows Preview ===")
    n = min(full.n, len(ds)) if len(ds) else 0
    for i in range(n):
        row = ds[i]
        keys = list(row.keys())
        print(f"[{i}] keys:", keys)
        # 打印每列的简短内容
        for k in keys:
            print(f"  - {k}: {shorten(row[k])}")
        print()

    # 猜测文本列
    txt_col = guess_text_column(ds.column_names)
    if txt_col:
        print(f"Detected text-like column: '{txt_col}'")
        print("Sample text:", shorten(ds[0][txt_col]) if len(ds) else "<empty>")
        print()
    else:
        print("No obvious text-like column found among", ds.column_names)
        print()

    # 如果存在 messages（聊天数据）
    if "messages" in ds.column_names:
        print("=== Detected 'messages' (chat format) ===")
        msgs = ds[0]["messages"] if len(ds) else None
        if isinstance(msgs, list):
            print(f"messages[0] sample length: {len(msgs)}")
            if msgs:
                print("First message:", shorten(msgs[0]))
        else:
            print("messages sample:", shorten(msgs))
        print()


def count_missing(ds: Dataset, col: str, limit: int = 20000):
    """粗略统计前 limit 条里该列缺失的比例，避免整库遍历太慢。"""
    missing = 0
    total = min(limit, len(ds))
    for i in range(total):
        v = ds[i].get(col, None)
        if v is None:
            missing += 1
    return missing, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="HF repo id（如 timdettmers/openassistant-guanaco）或本地 load_from_disk 路径")
    parser.add_argument("--split", default=None, help="要查看的 split（默认优先 train）")
    parser.add_argument("--n", type=int, default=3, help="预览样本条数")
    parser.add_argument("--check-col", default=None, help="额外检查某个列是否存在/缺失")
    args = parser.parse_args()

    ds, full = try_load_dataset(args.data, args.split)
    print_basic_info(ds, args)

    if args.check_col:
        if args.check_col in ds.column_names:
            miss, tot = count_missing(ds, args.check_col)
            print(f"Column '{args.check_col}' exists. Missing in first {tot} rows: {miss}")
        else:
            print(f"Column '{args.check_col}' NOT in dataset columns: {ds.column_names}")

    print("Done.")


if __name__ == "__main__":
    main()