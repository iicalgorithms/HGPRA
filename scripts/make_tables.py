#!/usr/bin/env python3
"""Render collected HGPRA CSV summaries as LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def _escape_latex(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _group_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        row.get("dataset", ""),
        row.get("ratio", ""),
        row.get("method", ""),
        row.get("variant", ""),
    )


def render_latex_table(
    rows: Iterable[Dict[str, str]],
    caption: str = "Generated experiment summary.",
    label: str = "tab:generated_results",
) -> str:
    grouped: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        value = row.get("test_acc_percent", "")
        if value == "":
            continue
        grouped[_group_key(row)].append(float(value))

    lines = [
        "\\begin{table}[!t]",
        "\\caption{" + _escape_latex(caption) + "}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llllc}",
        "\\toprule",
        "Dataset & Ratio & Method & Variant & Accuracy (\\%) \\\\",
        "\\midrule",
    ]
    for key in sorted(grouped):
        values = grouped[key]
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        dataset, ratio, method, variant = key
        lines.append(
            "{} & {} & {} & {} & {:.2f} $\\pm$ {:.2f} \\\\".format(
                _escape_latex(dataset),
                _escape_latex(ratio),
                _escape_latex(method),
                _escape_latex(variant),
                mean,
                std,
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\label{" + label + "}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def read_rows(input_path: str) -> List[Dict[str, str]]:
    with open(input_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/summary.csv")
    parser.add_argument("--out", default="results/generated_table.tex")
    parser.add_argument("--caption", default="Generated experiment summary.")
    parser.add_argument("--label", default="tab:generated_results")
    args = parser.parse_args()

    table = render_latex_table(read_rows(args.input), caption=args.caption, label=args.label)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(table)
    print(f"Wrote LaTeX table to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
