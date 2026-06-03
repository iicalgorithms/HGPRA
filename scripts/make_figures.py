#!/usr/bin/env python3
"""Render simple result figures from collected HGPRA CSV summaries."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List


def read_rows(input_path: str) -> List[Dict[str, str]]:
    with open(input_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_line_figure(
    rows: List[Dict[str, str]],
    x_field: str,
    y_field: str,
    group_field: str,
    out_path: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for figure generation. Install it or run make_tables.py only.") from exc

    grouped = defaultdict(list)
    for row in rows:
        if row.get(x_field, "") == "" or row.get(y_field, "") == "":
            continue
        grouped[row.get(group_field, "all")].append((float(row[x_field]), float(row[y_field])))

    if not grouped:
        raise SystemExit(f"No plottable rows found for x={x_field!r}, y={y_field!r}")

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for group, points in sorted(grouped.items()):
        points = sorted(points)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=group)
    ax.set_xlabel(xlabel or x_field.replace("_", " "))
    ax.set_ylabel(ylabel or y_field.replace("_", " "))
    ax.grid(True, linewidth=0.4, alpha=0.4)
    if len(grouped) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/summary.csv")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--x-field", default="beta")
    parser.add_argument("--y-field", default="test_acc_percent")
    parser.add_argument("--group-field", default="dataset")
    parser.add_argument("--name", default="sensitivity.png")
    parser.add_argument("--xlabel", default=None)
    parser.add_argument("--ylabel", default=None)
    args = parser.parse_args()

    out_path = os.path.join(args.out_dir, args.name)
    make_line_figure(
        read_rows(args.input),
        args.x_field,
        args.y_field,
        args.group_field,
        out_path,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
    )
    print(f"Wrote figure to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
