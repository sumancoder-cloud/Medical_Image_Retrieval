"""Print and save a readable report from retrieval evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_report(metrics_dir: Path, query_limit: int) -> str:
    summary = json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))
    classes = read_csv(metrics_dir / "per_class.csv")
    queries = read_csv(metrics_dir / "query_results.csv")
    lines = [
        "MEDICAL IMAGE RETRIEVAL REPORT",
        "=" * 32,
        f"Queries: {summary['queries']}",
        f"Database images: {summary['database_images']}",
        f"Model: {summary['model']}",
        f"Embedding dimension: {summary['embedding_dimension']}",
        f"Similarity: {summary['similarity']}",
        "",
        "OVERALL METRICS",
        "-" * 16,
        f"Precision@10: {float(summary['precision_at_10_macro_queries']):.6f}",
        f"Precision@20: {float(summary['precision_at_20_macro_queries']):.6f}",
        f"Precision@30: {float(summary['precision_at_30_macro_queries']):.6f}",
        f"Recall@30:    {float(summary['recall_at_30_macro_queries']):.6f}",
        "",
        "PER-CLASS METRICS",
        "-" * 18,
        "class  queries  P@10      P@20      P@30      R@30",
    ]
    for row in classes:
        lines.append(
            f"{row['class']:5}  {row['queries']:7}  "
            f"{float(row['precision_at_10']):.6f}  "
            f"{float(row['precision_at_20']):.6f}  "
            f"{float(row['precision_at_30']):.6f}  "
            f"{float(row['recall_at_30']):.6f}"
        )
    lowest = sorted(queries, key=lambda row: float(row["precision_at_10"]))[:query_limit]
    lines.extend(["", f"LOWEST P@10 QUERIES (first {query_limit})", "-" * 30])
    for row in lowest:
        lines.append(
            f"P@10={float(row['precision_at_10']):.2f} "
            f"P@20={float(row['precision_at_20']):.2f} "
            f"P@30={float(row['precision_at_30']):.2f} "
            f"class_id={row['query_class_id']} "
            f"path={row['query_path']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/paper_split/metrics"))
    parser.add_argument("--query-limit", type=int, default=10)
    args = parser.parse_args()
    report = make_report(args.metrics_dir, args.query_limit)
    output = args.metrics_dir / "report.txt"
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved readable report to {output}")


if __name__ == "__main__":
    main()
