#!/usr/bin/env python3
"""Build the paper sensitivity CSV from collected formal HGPRA summaries."""

from __future__ import annotations

import argparse
import csv
import os
from decimal import Decimal
from typing import Dict, Iterable, List, Tuple


DATASET_LABELS = {
    "20newsW100": "20News",
    "ModelNet40": "ModelNet40",
}

FIELDNAMES = ["dataset", "ratio", "beta", "test_acc_percent", "test_std_percent", "source_log"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str, rows: Iterable[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _decimal_key(value: str) -> Decimal:
    return Decimal(value)


def build_rows(summary_rows: Iterable[Dict[str, str]], ratio: str = "0.01") -> List[Dict[str, str]]:
    lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in summary_rows:
        if row.get("method") != "AllDeepSets":
            continue
        if row.get("ratio") != ratio:
            continue
        dataset = DATASET_LABELS.get(row.get("dataset", ""))
        beta = row.get("beta", "")
        if not dataset or beta == "":
            continue
        if not row.get("test_acc_percent"):
            continue
        lookup[(dataset, beta)] = {
            "dataset": dataset,
            "ratio": ratio,
            "beta": beta,
            "test_acc_percent": row["test_acc_percent"],
            "test_std_percent": row.get("test_std_percent", ""),
            "source_log": row.get("log_path", ""),
        }

    return [lookup[key] for key in sorted(lookup, key=lambda item: (item[0], _decimal_key(item[1])))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Collected formal summary CSV. Pass multiple times to merge main and sensitivity runs.",
    )
    parser.add_argument("--ratio", default="0.01", help="Condensation ratio used for the sensitivity plot.")
    parser.add_argument(
        "--out",
        default=os.path.join(CODE_DIR, "results", "paper", "sensitivity_beta.csv"),
        help="Output CSV consumed by make_figures.py.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []
    for input_path in args.input:
        rows.extend(_read_csv(input_path))
    output_rows = build_rows(rows, ratio=args.ratio)
    _write_csv(args.out, output_rows)
    print(f"Wrote {len(output_rows)} sensitivity rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
