#!/usr/bin/env python3
"""Collect HGPRA training logs into a CSV summary."""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as _dt
import os
import re
from typing import Dict, Iterable, List


DISTILL_DIR_RE = re.compile(r"(?P<dataset>.+)-reduce_(?P<ratio>[0-9.]+)-")
ARG_RE = re.compile(r"(\w+)=('[^']*'|True|False|None|-?[0-9.]+)")
NEW_BEST_RE = re.compile(r"test_acc = (?P<acc>[0-9.]+), iteration = (?P<iter>[0-9]+)")
EVAL_ACC_RE = re.compile(r"Evaluation ACC: (?P<method>\S+) best test_acc = (?P<acc>[0-9.]+), best_iter = (?P<iter>[0-9]+)")
MEAN_ACC_RE = re.compile(r"TEST: Full Graph Mean Accuracy: (?P<acc>[0-9.]+), STD: (?P<std>[0-9.]+)")
ITERATION_RE = re.compile(r"Iteration (?P<iter>[0-9]+):")
EARLY_STOP_RE = re.compile(r"Early-stop distill\(\)")
TIME_RE = re.compile(r"^(?P<stamp>[0-9]{2}/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} [AP]M)")
ISO_TIME_RE = re.compile(r"^(?P<stamp>[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})(?:,[0-9]+)?")


FIELDNAMES = [
    "dataset",
    "ratio",
    "preset",
    "variant",
    "method",
    "seed",
    "beta",
    "expanding_window",
    "test_acc_percent",
    "test_std_percent",
    "best_iter",
    "elapsed_seconds",
    "log_path",
]


def _parse_args_line(line: str) -> Dict[str, str]:
    if "Namespace(" not in line:
        return {}
    parsed: Dict[str, str] = {}
    for key, value in ARG_RE.findall(line):
        try:
            parsed[key] = str(ast.literal_eval(value))
        except (SyntaxError, ValueError):
            parsed[key] = value
    return parsed


def _infer_from_path(log_path: str) -> Dict[str, str]:
    parts = os.path.normpath(log_path).split(os.sep)
    inferred = {"dataset": "", "ratio": "", "preset": "", "variant": ""}
    if "Distill" in parts:
        idx = parts.index("Distill")
        if idx >= 2:
            inferred["preset"] = parts[idx - 2]
            inferred["variant"] = parts[idx - 1]
        if idx + 1 < len(parts):
            match = DISTILL_DIR_RE.match(parts[idx + 1])
            if match:
                inferred["dataset"] = match.group("dataset")
                inferred["ratio"] = match.group("ratio")
    return inferred


def _parse_time(line: str):
    match = TIME_RE.match(line)
    if match:
        return _dt.datetime.strptime(match.group("stamp"), "%m/%d %I:%M:%S %p")
    match = ISO_TIME_RE.match(line)
    if match:
        return _dt.datetime.strptime(match.group("stamp"), "%Y-%m-%d %H:%M:%S")
    return None


def _is_best_or_tied(value: float, best_value) -> bool:
    return best_value is None or value > best_value or abs(value - best_value) < 1e-9


def parse_log_file(log_path: str) -> Dict[str, str]:
    row = {field: "" for field in FIELDNAMES}
    row.update(_infer_from_path(log_path))
    row["log_path"] = log_path
    first_time = None
    last_time = None
    best_percent = None
    best_iter = ""
    best_std_percent = ""
    target_iteration = None
    last_iteration = -1
    completed = False

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stamp = _parse_time(line)
            if stamp is not None:
                first_time = first_time or stamp
                last_time = stamp

            args = _parse_args_line(line)
            if "ITER" in args:
                try:
                    target_iteration = int(float(args["ITER"]))
                except ValueError:
                    target_iteration = None
            for key in ("dname", "method", "seed", "beta", "expanding_window"):
                if key in args:
                    output_key = "dataset" if key == "dname" else key
                    row[output_key] = args[key]

            if EARLY_STOP_RE.search(line):
                completed = True

            match = ITERATION_RE.search(line)
            if match:
                last_iteration = max(last_iteration, int(match.group("iter")))

            match = NEW_BEST_RE.search(line)
            if match:
                value = float(match.group("acc"))
                if _is_best_or_tied(value, best_percent):
                    best_percent = value
                    best_iter = match.group("iter")

            match = EVAL_ACC_RE.search(line)
            if match:
                value = float(match.group("acc")) * 100.0
                row["method"] = row["method"] or match.group("method")
                if _is_best_or_tied(value, best_percent):
                    best_percent = value
                    best_iter = match.group("iter")

            match = MEAN_ACC_RE.search(line)
            if match:
                value = float(match.group("acc")) * 100.0
                if best_percent is None or value > best_percent:
                    best_percent = value
                    best_std_percent = f"{float(match.group('std')) * 100.0:.6f}"

    if best_percent is not None:
        row["test_acc_percent"] = f"{best_percent:.6f}"
    row["best_iter"] = best_iter
    row["test_std_percent"] = best_std_percent
    if first_time is not None and last_time is not None:
        if last_time < first_time:
            last_time += _dt.timedelta(days=1)
        row["elapsed_seconds"] = str(int((last_time - first_time).total_seconds()))
    if target_iteration is not None and last_iteration >= target_iteration:
        completed = True
    row["_completed"] = "True" if completed else "False"
    return row


def iter_train_logs(log_root: str) -> Iterable[str]:
    for root, _, files in os.walk(log_root):
        if "train.log" in files and "Distill" in os.path.normpath(root).split(os.sep):
            yield os.path.join(root, "train.log")


def collect_rows(log_root: str, complete_only: bool = False) -> List[Dict[str, str]]:
    rows = [parse_log_file(path) for path in sorted(iter_train_logs(log_root))]
    if complete_only:
        rows = [row for row in rows if row.get("_completed") == "True"]
    return rows


def latest_per_config(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    latest: Dict[tuple, Dict[str, str]] = {}
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("ratio", ""),
            row.get("preset", ""),
            row.get("variant", ""),
            row.get("method", ""),
            row.get("seed", ""),
            row.get("beta", ""),
            row.get("expanding_window", ""),
        )
        if key not in latest or row.get("log_path", "") > latest[key].get("log_path", ""):
            latest[key] = row
    return [latest[key] for key in sorted(latest)]


def write_csv(rows: Iterable[Dict[str, str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", default="logs_HGPRA")
    parser.add_argument("--out", default="results/summary.csv")
    parser.add_argument("--latest-per-config", action="store_true")
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Ignore distillation logs that have not reached ITER or an early-stop marker.",
    )
    args = parser.parse_args()

    rows = collect_rows(args.log_root, complete_only=args.complete_only)
    if args.latest_per_config:
        rows = latest_per_config(rows)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
