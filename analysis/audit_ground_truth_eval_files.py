#!/usr/bin/env python3
"""
Scan experiment output trees for GroundTruthAbstentionEvaluator.json.

Expected layout (under each sweep root):
  <sweep_root>/<run_id>/results/<dataset_dir>/<timestamp>/GroundTruthAbstentionEvaluator.json

Run IDs are any subdirectory name that contains a ``results`` child (not only numeric).

Example (from repo root, data on a mounted volume):
  uv run python analysis/audit_ground_truth_eval_files.py \\
    --sweep-glob '/path/to/data/Qwen2_5_*_coeff_*_v3_sweep' \\
    -o missing_evals.csv

  uv run python analysis/audit_ground_truth_eval_files.py \\
    --sweep-parent /path/to/data \\
    --sweep-name-glob '*coeff_*_v3_sweep' \\
    -o audit.csv --summary
"""

from __future__ import annotations

import argparse
import csv
import glob as stdglob
import sys
from pathlib import Path

EVAL_FILENAME = "GroundTruthAbstentionEvaluator.json"


def _has_eval_under_dataset(dataset_dir: Path) -> tuple[bool, str]:
    """
    Returns (found, relative_path_or_reason).
    Looks for EVAL_FILENAME in immediate subdirs of dataset_dir (timestamp runs).
    """
    if not dataset_dir.is_dir():
        return False, "not_a_directory"
    try:
        children = list[Path](dataset_dir.iterdir())
    except OSError as e:
        return False, f"list_error:{e}"

    if not children:
        return False, "empty_dataset_dir"

    for child in sorted(children):
        if child.is_dir():
            candidate = child / EVAL_FILENAME
            if candidate.is_file():
                return True, str(candidate)

    return False, "no_timestamp_dir_with_file"


def iter_sweep_roots(parent: Path | None, name_glob: str | None, globs: list[str]) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set[Path]()

    for g in globs:
        for s in sorted(stdglob.glob(g)):
            p = Path(s)
            rp = p.resolve()
            if rp.is_dir() and rp not in seen:
                seen.add(rp)
                roots.append(rp)

    if parent is not None and name_glob:
        p = parent.resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"--sweep-parent is not a directory: {p}")
        for child in sorted(p.glob(name_glob)):
            if child.is_dir():
                rc = child.resolve()
                if rc not in seen:
                    seen.add(rc)
                    roots.append(rc)

    return roots


def audit_sweep(sweep_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    sweep_str = str(sweep_root)

    try:
        run_candidates = [p for p in sweep_root.iterdir() if p.is_dir()]
    except OSError as e:
        return [
            {
                "sweep_dir": sweep_str,
                "run_id": "",
                "dataset": "",
                "has_eval_json": "false",
                "eval_json_path": "",
                "detail": f"sweep_list_error:{e}",
            }
        ]

    for run_dir in sorted(run_candidates, key=lambda p: p.name):
        results_dir = run_dir / "results"
        if not results_dir.is_dir():
            rows.append(
                {
                    "sweep_dir": sweep_str,
                    "run_id": run_dir.name,
                    "dataset": "",
                    "has_eval_json": "false",
                    "eval_json_path": "",
                    "detail": "no_results_dir",
                }
            )
            continue

        try:
            dataset_dirs = [p for p in results_dir.iterdir() if p.is_dir()]
        except OSError as e:
            rows.append(
                {
                    "sweep_dir": sweep_str,
                    "run_id": run_dir.name,
                    "dataset": "",
                    "has_eval_json": "false",
                    "eval_json_path": "",
                    "detail": f"results_list_error:{e}",
                }
            )
            continue

        if not dataset_dirs:
            rows.append(
                {
                    "sweep_dir": sweep_str,
                    "run_id": run_dir.name,
                    "dataset": "",
                    "has_eval_json": "false",
                    "eval_json_path": "",
                    "detail": "empty_results_dir",
                }
            )
            continue

        for ds in sorted(dataset_dirs, key=lambda p: p.name):
            ok, info = _has_eval_under_dataset(ds)
            rows.append(
                {
                    "sweep_dir": sweep_str,
                    "run_id": run_dir.name,
                    "dataset": ds.name,
                    "has_eval_json": "true" if ok else "false",
                    "eval_json_path": info if ok else "",
                    "detail": "" if ok else info,
                }
            )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-glob",
        action="append",
        default=[],
        metavar="GLOB",
        help="Glob for sweep root dirs (repeatable). Resolved from the current working directory.",
    )
    parser.add_argument(
        "--sweep-parent",
        type=Path,
        default=None,
        help="Directory whose immediate children are sweep roots.",
    )
    parser.add_argument(
        "--sweep-name-glob",
        default=None,
        help="With --sweep-parent, basename pattern for sweep dirs (e.g. '*coeff_*_v3_sweep').",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only write rows where has_eval_json is false.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print per-sweep pass/fail counts to stderr.",
    )
    args = parser.parse_args()

    if not args.sweep_glob and (args.sweep_parent is None or not args.sweep_name_glob):
        parser.error(
            "Provide --sweep-glob one or more times, or both --sweep-parent and --sweep-name-glob."
        )

    sweep_roots = iter_sweep_roots(args.sweep_parent, args.sweep_name_glob, args.sweep_glob)
    if not sweep_roots:
        print("No sweep directories matched.", file=sys.stderr)
        return 1

    all_rows: list[dict[str, str]] = []
    for root in sweep_roots:
        all_rows.extend(audit_sweep(root))

    if args.summary:
        from collections import defaultdict

        by_sweep: dict[str, list[dict[str, str]]] = defaultdict[str, list[dict[str, str]]](list)
        for r in all_rows:
            by_sweep[r["sweep_dir"]].append(r)

        for sweep, srows in sorted(by_sweep.items()):
            n_ok = sum(1 for r in srows if r["has_eval_json"] == "true")
            n_tot = len(srows)
            bad = n_tot - n_ok
            status = "PASS" if bad == 0 else f"FAIL ({bad} missing)"
            print(f"  {status}: {sweep}  ({n_ok}/{n_tot} ok)", file=sys.stderr)

    out_rows = all_rows
    if args.only_missing:
        out_rows = [r for r in all_rows if r["has_eval_json"] != "true"]

    fieldnames = [
        "sweep_dir",
        "run_id",
        "dataset",
        "has_eval_json",
        "eval_json_path",
        "detail",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
