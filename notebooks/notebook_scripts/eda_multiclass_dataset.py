"""
EDA script for abstention classifier datasets (eight_class / seven_class).
Run via 'python eda_multiclass_dataset.py'
Assumes CSV files have at minimum two columns: "sentence" and "cluster_name".
"""

import textwrap
from pathlib import Path
import pandas as pd

# Config
DATA_ROOT = Path("../../data/abstention_classifier_datasets")

LABEL_META = {
    "A": "Categorical Refusal",
    "B": "Epistemic Deflection",
    "C": "Insufficient Information",
    "D": "Context-Faithful Answer",
    "E": "Direct Factual Answer",
    "F": "Hallucination/Confabulation",
    "G": "Adversarial/Format",
    "H": "Composite",
}

SPLITS = ["train", "val", "test"]

SEP  = "=" * 72
SEP2 = "-" * 72


def sep(title: str = "") -> None:
    print(f"\n{SEP}")
    if title:
        print(f"  {title}")
        print(SEP2)


def load_all() -> dict:
    data = {}
    for variant in ["eight_class", "seven_class"]:
        data[variant] = {}
        for split in SPLITS:
            path = DATA_ROOT / variant / f"{split}.csv"
            data[variant][split] = pd.read_csv(path)
    return data


# 1. Schema
def inspect_schema(data: dict) -> None:
    sep("1. SCHEMA — columns, dtypes, nulls")
    for variant, splits in data.items():
        df = splits["train"]
        print(f"\n[{variant}] columns : {list(df.columns)}")
        print(f"[{variant}] dtypes  :\n{df.dtypes.to_string()}")
        nulls = df.isnull().sum()
        print(f"[{variant}] nulls   :\n{nulls.to_string()}")
        if nulls.any():
            print("  *** NULL ROWS ***")
            print(df[df.isnull().any(axis=1)].head(3).to_string())


# 2. Label encoding
def inspect_labels(data: dict) -> None:
    sep("2. LABEL ENCODING — unique values, dtype, unexpected values")
    expected_8 = set("ABCDEFGH")
    expected_7 = set("ABCDEFG")
    for variant, splits in data.items():
        df   = splits["train"]
        vals = sorted(df["cluster_name"].unique().tolist())
        print(f"\n[{variant}] unique labels : {vals}")
        print(f"[{variant}] label dtype   : {df['cluster_name'].dtype}")
        expected = expected_8 if variant == "eight_class" else expected_7
        unexpected = set(str(v) for v in vals) - expected
        if unexpected:
            print(f"  *** UNEXPECTED LABEL VALUES: {unexpected} ***")
        else:
            print(f"  OK — all labels in expected set {sorted(expected)}")


# 3. Split sizes & class counts
def inspect_counts(data: dict) -> None:
    sep("3. SPLIT SIZES & CLASS COUNTS")
    for variant, splits in data.items():
        print(f"\n[{variant}]")
        total = sum(len(splits[s]) for s in SPLITS)
        for split in SPLITS:
            df     = splits[split]
            counts = df["cluster_name"].value_counts().sort_index()
            print(f"\n  {split:5s}  n={len(df):>8,}  ({len(df)/total*100:.1f}% of all data)")
            for lbl, n in counts.items():
                name = LABEL_META.get(str(lbl), "?")
                bar  = "█" * int(n / len(df) * 40)
                print(f"    {lbl}  {n:>7,}  {n/len(df)*100:5.1f}%  {bar}  {name}")


# 4. Imbalance ratios
def inspect_imbalance(data: dict) -> None:
    sep("4. IMBALANCE — majority/minority ratios")
    for variant, splits in data.items():
        df     = splits["train"]
        counts = df["cluster_name"].value_counts()
        ratio  = counts.max() / counts.min()
        print(f"\n[{variant}] train imbalance ratio  : {ratio:.1f}x")
        print(f"[{variant}] majority class         : {counts.idxmax()}  ({counts.max():,}  {counts.max()/len(df)*100:.1f}%)")
        print(f"[{variant}] minority class         : {counts.idxmin()}  ({counts.min():,}  {counts.min()/len(df)*100:.1f}%)")
        if ratio > 20:
            print("  *** SEVERE IMBALANCE — strongly consider class weights or oversampling ***")
        elif ratio > 5:
            print("  *** MODERATE IMBALANCE — worth noting ***")


# 5. Stratification check
def inspect_stratification(data: dict) -> None:
    sep("5. STRATIFICATION — are class proportions consistent across splits?")
    for variant, splits in data.items():
        print(f"\n[{variant}]")
        rows = {}
        for split in SPLITS:
            df      = splits[split]
            counts  = df["cluster_name"].value_counts(normalize=True).sort_index() * 100
            rows[split] = counts
        table = pd.DataFrame(rows).round(1)
        table.index.name = "cluster_name"
        print(table.to_string())

        # Flag any label where max-min spread across splits > 3 pp
        spread = table.max(axis=1) - table.min(axis=1)
        bad    = spread[spread > 3]
        if not bad.empty:
            print(f"\n  *** Labels with >3pp spread across splits: {bad.to_dict()} ***")
        else:
            print("\n  OK — all labels within 3pp across splits")


# 6. Text length
def inspect_text_length(data: dict) -> None:
    sep("6. TEXT LENGTH — chars per label (train only)")
    for variant, splits in data.items():
        df = splits["train"].copy()
        df["len"] = df["sentence"].str.len()
        print(f"\n[{variant}]  overall: "
              f"median={df['len'].median():.0f}  "
              f"mean={df['len'].mean():.0f}  "
              f"p95={df['len'].quantile(0.95):.0f}  "
              f"max={df['len'].max():,}")
        stats = (
            df.groupby("cluster_name")["len"]
            .agg(n="count", median="median", mean="mean", p95=lambda x: x.quantile(0.95), max="max")
            .round(0)
            .astype(int)
        )
        print(stats.to_string())
        if df["len"].median() > 512:
            print("  *** Median > 512 chars — check tokeniser truncation in AutoML config ***")


# 7. Duplicate & cross-split leak detection
def inspect_duplicates(data: dict) -> None:
    sep("7. DUPLICATES & CROSS-SPLIT LEAKAGE")
    for variant, splits in data.items():
        train = splits["train"]
        val   = splits["val"]
        test  = splits["test"]

        # Exact duplicates within train
        n_dup = train.duplicated(subset=["sentence"]).sum()
        print(f"\n[{variant}] exact text dupes within train : {n_dup:,}  ({n_dup/len(train)*100:.2f}%)")

        # Per-label duplicate rate (paper warns stereotyped refusals repeat verbatim)
        dup_by_label = (
            train.groupby("cluster_name")
            .apply(lambda g: g.duplicated(subset=["sentence"]).sum())
            .rename("dupes")
        )
        print(f"[{variant}] dupes per label (train):")
        print(dup_by_label.to_string())

        # Cross-split leakage: train texts appearing in val/test
        train_texts = set(train["sentence"])
        val_leak    = splits["val"]["sentence"].isin(train_texts).sum()
        test_leak   = splits["test"]["sentence"].isin(train_texts).sum()
        print(f"[{variant}] train→val  leak : {val_leak:,}  ({val_leak/len(splits['val'])*100:.2f}%)")
        print(f"[{variant}] train→test leak : {test_leak:,}  ({test_leak/len(splits['test'])*100:.2f}%)")
        if val_leak > 0 or test_leak > 0:
            print("  *** LEAKAGE DETECTED — evaluation metrics will be inflated ***")


# 8. 7-class vs 8-class relationship
def inspect_variant_diff(data: dict) -> None:
    sep("8. SEVEN-CLASS vs EIGHT-CLASS — is it exactly H-dropped, or more?")
    for split in SPLITS:
        df8 = data["eight_class"][split]
        df7 = data["seven_class"][split]

        # Is seven_class a strict subset (H removed)?
        df8_no_h    = df8[df8["cluster_name"] != "H"].reset_index(drop=True)
        sizes_match = len(df8_no_h) == len(df7)

        print(f"\n  [{split}]")
        print(f"    eight_class rows         : {len(df8):,}")
        print(f"    seven_class rows         : {len(df7):,}")
        print(f"    eight_class minus H rows : {len(df8_no_h):,}")
        print(f"    sizes match after -H     : {sizes_match}")

        if sizes_match:
            # Check if the rows are identical (order-agnostic via text set)
            texts8 = set(df8_no_h["sentence"])
            texts7 = set(df7["sentence"])
            extra_in_7  = texts7 - texts8
            missing_in_7 = texts8 - texts7
            print(f"    rows in 7 not in 8-H    : {len(extra_in_7)}")
            print(f"    rows in 8-H not in 7    : {len(missing_in_7)}")
            if not extra_in_7 and not missing_in_7:
                print("    CONFIRMED: seven_class is exactly eight_class with H dropped")
            else:
                print("    *** DIFFERS beyond H removal — inspect further ***")
        else:
            print("    *** Row counts don't align — seven_class is NOT a simple H-drop ***")


# 9. Sample texts per label
def inspect_samples(data: dict, n_samples: int = 2) -> None:
    sep("9. SAMPLE TEXTS — qualitative check (eight_class train)")
    df = data["eight_class"]["train"]
    for lbl in sorted(df["cluster_name"].unique()):
        name    = LABEL_META.get(str(lbl), "?")
        samples = df[df["cluster_name"] == lbl]["sentence"].sample(
            min(n_samples, (df["cluster_name"] == lbl).sum()), random_state=42
        )
        print(f"\n  [{lbl}] {name}")
        for txt in samples:
            wrapped = textwrap.fill(txt[:400], width=68, initial_indent="    > ", subsequent_indent="      ")
            print(wrapped)
            if len(txt) > 400:
                print("      [truncated]")


# Main
if __name__ == "__main__":
    print(SEP)
    print("  ABSTENTION DATASET EDA")
    print(SEP)

    data = load_all()

    inspect_schema(data)
    inspect_labels(data)
    inspect_counts(data)
    inspect_imbalance(data)
    inspect_stratification(data)
    inspect_text_length(data)
    inspect_duplicates(data)
    inspect_variant_diff(data)
    inspect_samples(data)

    print(f"\n{SEP}\n  DONE\n{SEP}\n")