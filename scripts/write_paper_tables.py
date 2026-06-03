#!/usr/bin/env python3
"""Generate LaTeX table fragments used by the ICDM manuscript."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence


MAIN_METHODS = [
    "Random",
    "Herding",
    "K-Center",
    "HyperSF",
    "HyperEF",
    "HG-Cond",
    "HGPRA",
]

CROSS_ARCH_METHODS = ["Herding", "HyperSF", "HG-Cond", "HGPRA", "Full Training Set"]
CROSS_ARCH_BACKBONES = ["HyperGCN", "HGNN", "AllDeepSets", "AllSetTransformer"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(CODE_DIR)


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _pm(value: str) -> str:
    return value.strip().replace("±", "\\pm").replace("+-", "\\pm")


def _mean(value: str) -> float:
    match = re.match(r"\s*([0-9.]+)", value)
    if not match:
        return float("-inf")
    return float(match.group(1))


def _text(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _math(value: str, bold: bool = False) -> str:
    body = f"${_pm(value)}$"
    if bold:
        return f"\\textbf{{\\boldmath {body}}}"
    return body


def _write(path: str, lines: Sequence[str]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def write_main_performance_table(csv_path: str, out_path: str) -> None:
    rows = read_csv(csv_path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    lines = [
        "\\begin{table*}[!t]",
        "\\caption{Node classification accuracy (\\%) under different condensation ratios. ``Full training set'' denotes training on the original dataset. Boldface indicates the best condensed result in each configuration.}",
        "\\scriptsize",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l|c|ccccc|cc|c}",
        "\\toprule",
        "\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Ratio} & \\multicolumn{5}{c|}{Size Reduction Baselines} & \\multicolumn{2}{c|}{Hypergraph Condensation} & \\multirow{2}{*}{Full Training Set} \\\\ \\cline{3-9}",
        "                         &                        & Random & Herding & K-Center & HyperSF & HyperEF & HG-Cond & HGPRA & \\\\",
        "\\midrule",
    ]
    dataset_items = list(grouped.items())
    for dataset_index, (dataset, dataset_rows) in enumerate(dataset_items):
        dataset_rows = sorted(dataset_rows, key=lambda item: float(item["ratio"].rstrip("%")), reverse=True)
        full_value = dataset_rows[0]["Full Training Set"]
        for row_index, row in enumerate(dataset_rows):
            best_method = max(MAIN_METHODS, key=lambda method: _mean(row[method]))
            prefix = f"\\multirow{{{len(dataset_rows)}}}{{*}}{{{_text(dataset)}}}" if row_index == 0 else ""
            cells = [prefix, _text(row["ratio"])]
            for method in MAIN_METHODS:
                cells.append(_math(row[method], bold=(method == best_method)))
            if row_index == 0:
                cells.append(f"\\multirow{{{len(dataset_rows)}}}{{*}}{{{_math(full_value)}}}")
            else:
                cells.append("")
            lines.append(" & ".join(cells) + " \\\\")
        if dataset_index != len(dataset_items) - 1:
            lines.append("\\hline")
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\label{tab:performance_comparison}", "\\end{table*}"])
    _write(out_path, lines)


def write_cross_arch_table(csv_path: str, out_path: str) -> None:
    rows = read_csv(csv_path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["ratio"])].append(row)

    lines = [
        "\\begin{table*}[!t]",
        "\\caption{Cross-architecture evaluation on condensed data. Accuracy is reported in percent. Boldface indicates the best condensed result for each architecture within each dataset block.}",
        "\\scriptsize",
        "\\centering",
        "\\resizebox{0.7\\textwidth}{!}{",
        "\\begin{tabular}{lc|cccc}",
        "\\toprule",
        "\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Method} & \\multicolumn{4}{c}{Architecture} \\\\ \\cline{3-6}",
        "                         &                         & HyperGCN & HGNN & AllDeepSets & AllSetTransformer \\\\",
        "\\midrule",
    ]
    group_items = list(grouped.items())
    for group_index, ((dataset, ratio), group_rows) in enumerate(group_items):
        by_method = {row["method"]: row for row in group_rows}
        for method_index, method in enumerate(CROSS_ARCH_METHODS):
            row = by_method[method]
            prefix = (
                f"\\multirow{{{len(CROSS_ARCH_METHODS)}}}{{*}}{{{_text(dataset)} ($r={_text(ratio)}$)}}"
                if method_index == 0
                else ""
            )
            cells = [prefix, _text(method)]
            for backbone in CROSS_ARCH_BACKBONES:
                value = row[backbone]
                condensed = [by_method[m][backbone] for m in CROSS_ARCH_METHODS if m != "Full Training Set"]
                best_value = max(condensed, key=_mean)
                cells.append(f"\\textbf{{{value}}}" if value == best_value and method != "Full Training Set" else value)
            lines.append(" & ".join(cells) + " \\\\")
        if group_index != len(group_items) - 1:
            lines.append("\\hline")
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\label{tab:hypergnn_comparison}", "\\end{table*}"])
    _write(out_path, lines)


def write_ablation_table(csv_path: str, out_path: str) -> None:
    rows = read_csv(csv_path)
    if rows and "setting" in rows[0]:
        write_component_ablation_table(rows, out_path)
        return

    lines = [
        "\\begin{table}[!t]",
        "\\caption{Ablation study on the progressive anchoring strategy.}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{c|cc|cc}",
        "\\toprule",
        "\\multirow{2}{*}{Matching Range} & \\multicolumn{2}{c|}{Cora-CA} & \\multicolumn{2}{c}{Citeseer-CC} \\\\ \\cline{2-5}",
        "                                & w/o PaS & PaS & w/o PaS & PaS \\\\ \\hline",
    ]
    for row in rows:
        best_cora = max(_mean(row["cora_without_pas"]), _mean(row["cora_pas"]))
        best_citeseer = max(_mean(row["citeseer_without_pas"]), _mean(row["citeseer_pas"]))
        cells = [
            _text(row["matching_range"]),
            _math(row["cora_without_pas"], _mean(row["cora_without_pas"]) == best_cora and row["matching_range"] == "PRA"),
            _math(row["cora_pas"], _mean(row["cora_pas"]) == best_cora and row["matching_range"] == "PRA"),
            _math(row["citeseer_without_pas"], _mean(row["citeseer_without_pas"]) == best_citeseer and row["matching_range"] == "PRA"),
            _math(row["citeseer_pas"], _mean(row["citeseer_pas"]) == best_citeseer and row["matching_range"] == "PRA"),
        ]
        lines.append(" & ".join(cells) + " \\\\ \\hline")
    lines[-1] = lines[-1].replace(" \\\\ \\hline", " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\label{tab:ablation}", "\\end{table}"])
    _write(out_path, lines)


def write_component_ablation_table(rows: List[Dict[str, str]], out_path: str) -> None:
    dataset_columns = [column for column in rows[0].keys() if column != "setting"]
    lines = [
        "\\begin{table}[!t]",
        "\\caption{Component ablation of HGPRA under the default AllDeepSets backbone at a 3\\% condensation ratio.}",
        "\\centering",
        "\\resizebox{0.8\\linewidth}{!}{",
        "\\begin{tabular}{l" + "c" * len(dataset_columns) + "}",
        "\\toprule",
        "Setting & " + " & ".join(_text(column) for column in dataset_columns) + " \\\\",
        "\\midrule",
    ]
    best_by_dataset = {
        dataset: max((_mean(row[dataset]) for row in rows), default=float("-inf"))
        for dataset in dataset_columns
    }
    for row in rows:
        cells = [_text(row["setting"])]
        for dataset in dataset_columns:
            value = row[dataset]
            cells.append(_math(value, bold=_mean(value) == best_by_dataset[dataset]))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\label{tab:ablation}", "\\end{table}"])
    _write(out_path, lines)


def write_runtime_table(csv_path: str, out_path: str) -> None:
    rows = read_csv(csv_path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    lines = [
        "\\begin{table}[!t]",
        "\\caption{Wall-clock times for representative settings. HGPRA times are measured in the local CPU reproduction, while HG-Cond entries are retained from the baseline table and are not a same-hardware comparison.}",
        "\\centering",
        "\\resizebox{0.6\\linewidth}{!}{",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Dataset & Ratio & HG-Cond & HGPRA \\\\",
        "\\midrule",
    ]
    items = list(grouped.items())
    for dataset_index, (dataset, dataset_rows) in enumerate(items):
        for row_index, row in enumerate(dataset_rows):
            prefix = f"\\multirow{{{len(dataset_rows)}}}{{*}}{{{_text(dataset)}}}" if row_index == 0 else ""
            lines.append(f"{prefix} & {_text(row['ratio'])} & {_text(row['HG-Cond'])} & {_text(row['HGPRA'])} \\\\")
        if dataset_index != len(items) - 1:
            lines.append("\\hline")
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\label{tab:time}", "\\end{table}"])
    _write(out_path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.path.join(CODE_DIR, "results", "paper"))
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_DIR, "tables"))
    args = parser.parse_args()

    write_main_performance_table(
        os.path.join(args.results_dir, "main_performance.csv"),
        os.path.join(args.out_dir, "performance_comparison.tex"),
    )
    write_cross_arch_table(
        os.path.join(args.results_dir, "cross_architecture.csv"),
        os.path.join(args.out_dir, "hypergnn_comparison.tex"),
    )
    component_ablation_path = os.path.join(args.results_dir, "component_ablation.csv")
    ablation_path = component_ablation_path if os.path.exists(component_ablation_path) else os.path.join(args.results_dir, "ablation_pas_pra.csv")
    write_ablation_table(ablation_path, os.path.join(args.out_dir, "ablation.tex"))
    write_runtime_table(
        os.path.join(args.results_dir, "runtime.csv"),
        os.path.join(args.out_dir, "time.tex"),
    )
    print(f"Wrote paper table fragments to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
