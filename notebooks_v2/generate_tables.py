#!/usr/bin/env python3
"""Generate LaTeX result tables from data/*.csv.

Run with:
  AbstentionBench/.venv/bin/python generate_tables.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Vanilla is the baseline column; remaining methods appear in this order.
METHODS = ["LoRA", "DoRA", "DPO", "GRPO", "SPT-100", "HPT-FS", "CAA"]
COLUMN_ALIASES = {"STP-100": "SPT-100"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_ALIASES)


def generate_latex_table(
    df,
    label,
    caption,
    is_higher_better=True,
    is_percentage=True,
):
    ncol = 2 + len(METHODS) + 1  # model + baseline + methods + avg delta
    col_spec = "l" + "c" * (ncol - 1)

    latex = []
    latex.append("\\begin{table}[h!]")
    latex.append("\\centering")
    latex.append("\\small")
    latex.append("\\setlength{\\tabcolsep}{2pt}")
    latex.append("\\resizebox{\\columnwidth}{!}{%")
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    header = (
        "\\textbf{Model} & \\textbf{Baseline} & "
        + " & ".join(f"\\textbf{{{m}}}" for m in METHODS)
        + " & \\textbf{Avg. $\\Delta$} \\\\"
    )
    latex.append(header)
    latex.append("\\midrule")

    all_deltas = {m: [] for m in METHODS}
    avg_deltas_per_model = []

    for _, row in df.iterrows():
        model = row["Model"]
        baseline = row["Vanilla"]
        if is_percentage:
            baseline *= 100

        row_latex = [f"{model} & {baseline:.2f} &"]

        deltas = {}
        for m in METHODS:
            if m in row and not pd.isna(row[m]):
                val = row[m]
                if is_percentage:
                    val *= 100
                deltas[m] = val - baseline
            else:
                deltas[m] = None

        valid_deltas = {k: v for k, v in deltas.items() if v is not None}
        if not valid_deltas:
            continue

        if is_higher_better:
            best_method = max(valid_deltas, key=valid_deltas.get)
        else:
            best_method = min(valid_deltas, key=valid_deltas.get)

        avg_delta_model = np.mean(list(valid_deltas.values()))
        avg_deltas_per_model.append(avg_delta_model)

        for i, m in enumerate(METHODS):
            if deltas[m] is None:
                cell = "   "
            else:
                val = row[m]
                if is_percentage:
                    val *= 100
                delta = deltas[m]
                all_deltas[m].append(delta)

                if delta > 0:
                    color = "green!60!black" if is_higher_better else "red"
                    sign = "+"
                elif delta < 0:
                    color = "red" if is_higher_better else "green!60!black"
                    sign = ""
                else:
                    color = "black"
                    sign = "+"

                delta_str = f"\\textcolor{{{color}}}{{({sign}{delta:.2f})}}"
                if m == best_method:
                    delta_str = (
                        f"\\textcolor{{{color}}}{{\\underline{{({sign}{delta:.2f})}}}}"
                    )

                cell = (
                    f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{val:.2f}\\\\"
                    f"{delta_str}\\end{{tabular}}"
                )

            row_latex.append(cell)
            if i < len(METHODS) - 1:
                row_latex.append("&")

        sign_avg = "+" if avg_delta_model > 0 else ""
        row_latex.append(f"& \\textbf{{{sign_avg}{avg_delta_model:.2f}}} \\\\")

        if "7B" in model or "1B" in model:
            row_latex[-1] += " \\hline"

        latex.append(" ".join(row_latex))

    latex.append("\\midrule")

    baseline_avg = df["Vanilla"].mean()
    if is_percentage:
        baseline_avg *= 100

    avg_row = [f"\\textbf{{Avg. $\\Delta$}} & \\textbf{{{baseline_avg:.2f}}} &"]

    for i, m in enumerate(METHODS):
        if all_deltas[m]:
            mean_d = np.mean(all_deltas[m])
            sign = "+" if mean_d > 0 else ""
            cell = (
                f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}"
                f"\\textbf{{{sign}{mean_d:.2f}}}\\end{{tabular}}"
            )
        else:
            cell = "   "
        avg_row.append(cell)
        if i < len(METHODS) - 1:
            avg_row.append("&")

    total_avg = np.mean(avg_deltas_per_model)
    sign_total = "+" if total_avg > 0 else ""
    avg_row.append(f"& \\textbf{{{sign_total}{total_avg:.2f}}} \\\\")

    latex.append(" ".join(avg_row))
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}%")
    latex.append("}")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def aggregate_metric_csv(path: Path, model: str) -> dict:
    df = normalize_columns(pd.read_csv(path))
    row = {"Model": model}
    for col in ["Vanilla", *METHODS]:
        if col in df.columns:
            row[col] = df[col].mean()
    return row


def main():
    df_f1 = normalize_columns(pd.read_csv(DATA / "abstention" / "in_domain_f1.csv"))
    print("--- F1 ---")
    print(
        generate_latex_table(
            df_f1,
            "tab:abstentionbench_results",
            "AbstentionBench results (F1 scores) across models and alignment methods. "
            "Deltas relative to the unaligned baseline are shown in parentheses. "
            "For each model, the best-performing alignment method is underlined. "
            "The final column reports the mean $\\Delta$ across aligned methods for each model.",
            is_higher_better=True,
            is_percentage=False,
        )
    )

    df_ood = normalize_columns(pd.read_csv(DATA / "abstention" / "rulebreakers_f1.csv"))
    print("\n--- RULEBREAKERS ---")
    print(
        generate_latex_table(
            df_ood,
            "tab:rulebreakers_results",
            "RULEBREAKERS results (F1 scores) across models and alignment methods. "
            "Deltas relative to the unaligned baseline are shown in parentheses. "
            "For each model, the best-performing alignment method is underlined. "
            "The final column reports the mean $\\Delta$ across aligned methods for each model.",
            is_higher_better=True,
            is_percentage=False,
        )
    )

    qa_files = [
        (DATA / "general_cap" / "qa" / "qwen0_5_qa.csv", "Qwen 2.5 0.5B"),
        (DATA / "general_cap" / "qa" / "qwen1_5_qa.csv", "Qwen 2.5 1.5B"),
        (DATA / "general_cap" / "qa" / "qwen3_qa.csv", "Qwen 2.5 3B"),
        (DATA / "general_cap" / "qa" / "qwen7_qa.csv", "Qwen 2.5 7B"),
        (DATA / "general_cap" / "qa" / "tulu8_qa.csv", "Llama 3.1 Tulu 3.1 8B"),
    ]
    df_qa = pd.DataFrame([aggregate_metric_csv(path, model) for path, model in qa_files])
    print("\n--- QA ---")
    print(
        generate_latex_table(
            df_qa,
            "tab:qa_results",
            "Mean accuracy across eight reasoning benchmarks (Section~\\ref{cha:evaluation}). "
            "Deltas relative to the unaligned baseline are shown in parentheses. "
            "For each model, the best-performing alignment method (highest accuracy) is underlined. "
            "The final column reports the mean $\\Delta$ across aligned methods for each model.",
            is_higher_better=True,
            is_percentage=False,
        )
    )

    ppl_files = [
        (DATA / "general_cap" / "ppl" / "qwen0_5_ppl.csv", "Qwen 2.5 0.5B"),
        (DATA / "general_cap" / "ppl" / "qwen1_5_ppl.csv", "Qwen 2.5 1.5B"),
        (DATA / "general_cap" / "ppl" / "qwen3_ppl.csv", "Qwen 2.5 3B"),
        (DATA / "general_cap" / "ppl" / "qwen7_ppl.csv", "Qwen 2.5 7B"),
        (DATA / "general_cap" / "ppl" / "gemma1_ppl.csv", "Gemma 3 1B"),
        (DATA / "general_cap" / "ppl" / "tulu8_ppl.csv", "Llama 3.1 Tulu 3.1 8B"),
    ]
    df_ppl = pd.DataFrame([aggregate_metric_csv(path, model) for path, model in ppl_files])
    print("\n--- PPL ---")
    print(
        generate_latex_table(
            df_ppl,
            "tab:ppl_results",
            "Mean language-modelling perplexity (PPL) on FLORES+ across English, Arabic, "
            "German, Spanish, Hindi, Russian and Chinese. Deltas relative to the unaligned "
            "baseline are shown in parentheses. For each model, the best-performing alignment "
            "method (lowest perplexity, hence lowest or negative delta) is underlined. "
            "The final column reports the mean $\\Delta$ across aligned methods for each model.",
            is_higher_better=False,
            is_percentage=False,
        )
    )


if __name__ == "__main__":
    main()
