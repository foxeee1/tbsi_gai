#!/usr/bin/env python3
"""Build a static blind image index without scores or model outputs."""
import argparse
import csv
import html
import os
from pathlib import Path
from urllib.parse import quote


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Output exists; refuse overwrite")
    with args.review.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    images = [path for path in args.images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS]
    matches = {}
    missing = []
    for row in rows:
        event_id = row["event_id"]
        hits = [path for path in images if event_id.lower() in path.stem.lower()]
        if not hits:
            missing.append(event_id)
        elif len(hits) > 4:
            raise ValueError(f"{event_id}: {len(hits)} images; ambiguous mapping")
        else:
            matches[event_id] = sorted(hits)
    if missing:
        raise ValueError(
            f"Images absent for {len(missing)} events, examples: {missing[:8]}; inspect naming/mapping"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html><html lang='zh'><meta charset='utf-8'>",
        "<title>盲审图像索引</title><style>body{font:16px sans-serif;max-width:1100px;margin:auto}article{margin:24px 0;padding:12px;border:1px solid #aaa}img{max-width:100%;max-height:600px;margin:6px}small{color:#666}</style>",
        "<h1>盲审图像索引</h1><p>逐张查看候选图像，并在独立 CSV 中填写标签。本页不提供自动分数、baseline 预测或干预结果。若图片缺少候选定位或含有信息泄漏，应停止审阅。</p>",
    ]
    for row in rows:
        event_id = html.escape(row["event_id"])
        parts.append(f"<article id='{event_id}'><h2>{event_id}</h2><small>{html.escape(row['seq'])} / frame {html.escape(row['frame'])}</small><br>")
        for path in matches[row["event_id"]]:
            rel = Path(os.path.relpath(path, args.out.parent)).as_posix()
            src = quote(rel, safe="/.-_~")
            parts.append(f"<a href='{src}'><img loading='lazy' src='{src}' alt='{event_id}'></a>")
        parts.append("</article>")
    parts.append("</html>")
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Indexed {len(rows)} events / {sum(map(len, matches.values()))} images; no model results read")


if __name__ == "__main__":
    main()
