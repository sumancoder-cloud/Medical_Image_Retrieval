"""Evaluate cosine image retrieval and write auditable metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def precision_at_k(retrieved_labels: np.ndarray, query_label: int, k: int) -> float:
    return float(np.mean(retrieved_labels[:k] == query_label))


def evaluate(features: np.ndarray, labels: np.ndarray, splits: np.ndarray, paths: np.ndarray) -> tuple[list[dict], list[dict]]:
    database = np.flatnonzero(splits == "database")
    queries = np.flatnonzero(splits == "test")
    if not len(database) or not len(queries):
        raise ValueError("Both database and test rows are required.")
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    similarities = normalized[queries] @ normalized[database].T
    rows: list[dict] = []
    retrieval_rows: list[dict] = []
    for query_index, query_scores in zip(queries, similarities):
        order = np.argsort(-query_scores, kind="stable")
        ranked_labels = labels[database[order]]
        query_label = int(labels[query_index])
        relevant_total = int(np.sum(labels[database] == query_label))
        row = {
            "query_path": str(paths[query_index]),
            "query_class_id": query_label,
            "database_relevant_count": relevant_total,
            "precision_at_10": precision_at_k(ranked_labels, query_label, 10),
            "precision_at_20": precision_at_k(ranked_labels, query_label, 20),
            "precision_at_30": precision_at_k(ranked_labels, query_label, 30),
            "recall_at_30": float(np.sum(ranked_labels[:30] == query_label) / relevant_total),
        }

        rows.append(row)
        for rank, database_position in enumerate(order[:30], start=1):
            database_index = database[database_position]
            retrieval_rows.append(
                {
                    "query_path": str(paths[query_index]),
                    "rank": rank,
                    "retrieved_path": str(paths[database_index]),
                    "retrieved_class_id": int(labels[database_index]),
                    "cosine_similarity": float(query_scores[database_position]),
                    "is_relevant": bool(labels[database_index] == query_label),
                }
            )
    return rows, retrieval_rows


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def per_class_metrics(rows: list[dict], classes: np.ndarray) -> list[dict]:
    result = []
    for class_id, class_name in enumerate(classes):
        selected = [row for row in rows if row["query_class_id"] == class_id]
        if not selected:
            continue
        result.append(
            {
                "class_id": class_id,
                "class": str(class_name),
                "queries": len(selected),
                "precision_at_10": mean(selected, "precision_at_10"),
                "precision_at_20": mean(selected, "precision_at_20"),
                "precision_at_30": mean(selected, "precision_at_30"),
                "recall_at_30": mean(selected, "recall_at_30"),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.features, allow_pickle=False)
    rows, retrieval_rows = evaluate(data["features"], data["labels"], data["splits"], data["paths"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "query_results.csv", rows)
    write_csv(args.output_dir / "top_retrievals.csv", retrieval_rows)
    class_rows = per_class_metrics(rows, data["classes"])
    write_csv(args.output_dir / "per_class.csv", class_rows)
    summary = {
        "queries": len(rows),
        "database_images": int(np.sum(data["splits"] == "database")),
        "precision_at_10_macro_queries": mean(rows, "precision_at_10"),
        "precision_at_20_macro_queries": mean(rows, "precision_at_20"),
        "precision_at_30_macro_queries": mean(rows, "precision_at_30"),
        "recall_at_30_macro_queries": mean(rows, "recall_at_30"),
        "model": str(data["model"][0]),
        "embedding_dimension": int(data["features"].shape[1]),
        "similarity": "cosine",
        "relevance": "same normalized class",
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
