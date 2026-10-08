import os
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

BASE = os.path.expanduser("~/Documents/qsarmil")
INP = os.path.join(
    BASE, "data", "esol_descriptor_architecture_interaction"
)
OUT = os.path.join(INP, "fdr_analysis")
os.makedirs(OUT, exist_ok=True)


def bh_correct(df, pcol, qcol):

    df = df.copy()

    valid = df[pcol].notna()

    df[qcol] = np.nan
    df["significant_fdr_0.05"] = False

    if valid.sum() > 0:

        reject, qvalues, _, _ = multipletests(
            df.loc[valid, pcol].to_numpy(),
            alpha=0.05,
            method="fdr_bh"
        )

        df.loc[valid, qcol] = qvalues
        df.loc[valid, "significant_fdr_0.05"] = reject

    return df


# ================================================================
# DESCRIPTOR COMPARISONS
# ================================================================

descriptor_file = os.path.join(
    INP,
    "paired_descriptor_comparisons.csv"
)

descriptor = pd.read_csv(descriptor_file)

descriptor = bh_correct(
    descriptor,
    "wilcoxon_p",
    "BH_q"
)

descriptor.to_csv(
    os.path.join(
        OUT,
        "paired_descriptor_comparisons_FDR.csv"
    ),
    index=False
)


# ================================================================
# ARCHITECTURE COMPARISONS
# ================================================================

architecture_file = os.path.join(
    INP,
    "paired_architecture_comparisons.csv"
)

architecture = pd.read_csv(architecture_file)

architecture = bh_correct(
    architecture,
    "wilcoxon_p",
    "BH_q"
)

architecture.to_csv(
    os.path.join(
        OUT,
        "paired_architecture_comparisons_FDR.csv"
    ),
    index=False
)


# ================================================================
# SUMMARY FUNCTION
# ================================================================

def summary_by_conformer(df, label):

    rows = []

    for nconf in sorted(
        df["n_conformers"].unique()
    ):

        sub = df[
            df["n_conformers"] == nconf
        ]

        valid = sub["wilcoxon_p"].notna()

        rows.append({
            "comparison_type": label,
            "n_conformers": nconf,
            "total_tests": len(sub),
            "valid_tests": int(valid.sum()),
            "raw_p_lt_0.05": int(
                (sub.loc[valid, "wilcoxon_p"] < 0.05).sum()
            ),
            "FDR_q_lt_0.05": int(
                (sub.loc[valid, "BH_q"] < 0.05).sum()
            ),
            "minimum_raw_p": sub["wilcoxon_p"].min(),
            "minimum_BH_q": sub["BH_q"].min(),
        })

    return pd.DataFrame(rows)


descriptor_summary = summary_by_conformer(
    descriptor,
    "descriptor"
)

architecture_summary = summary_by_conformer(
    architecture,
    "architecture"
)

summary = pd.concat(
    [
        descriptor_summary,
        architecture_summary
    ],
    ignore_index=True
)

summary.to_csv(
    os.path.join(
        OUT,
        "FDR_summary_by_conformer.csv"
    ),
    index=False
)


# ================================================================
# OVERALL SUMMARY
# ================================================================

overall = pd.DataFrame([
    {
        "comparison_type": "descriptor",
        "total_tests": len(descriptor),
        "raw_p_lt_0.05": int(
            (descriptor["wilcoxon_p"] < 0.05).sum()
        ),
        "FDR_q_lt_0.05": int(
            (descriptor["BH_q"] < 0.05).sum()
        ),
        "minimum_raw_p": descriptor["wilcoxon_p"].min(),
        "minimum_BH_q": descriptor["BH_q"].min(),
    },
    {
        "comparison_type": "architecture",
        "total_tests": len(architecture),
        "raw_p_lt_0.05": int(
            (architecture["wilcoxon_p"] < 0.05).sum()
        ),
        "FDR_q_lt_0.05": int(
            (architecture["BH_q"] < 0.05).sum()
        ),
        "minimum_raw_p": architecture["wilcoxon_p"].min(),
        "minimum_BH_q": architecture["BH_q"].min(),
    }
])

overall.to_csv(
    os.path.join(
        OUT,
        "FDR_overall_summary.csv"
    ),
    index=False
)


# ================================================================
# STRONGEST EFFECTS
# ================================================================

descriptor_top = descriptor.sort_values(
    "BH_q"
).head(30)

architecture_top = architecture.sort_values(
    "BH_q"
).head(30)

descriptor_top.to_csv(
    os.path.join(
        OUT,
        "top30_descriptor_effects_FDR.csv"
    ),
    index=False
)

architecture_top.to_csv(
    os.path.join(
        OUT,
        "top30_architecture_effects_FDR.csv"
    ),
    index=False
)


# ================================================================
# TERMINAL OUTPUT
# ================================================================

print("\n" + "=" * 80)
print("BENJAMINI-HOCHBERG FDR ANALYSIS")
print("=" * 80)

print("\nOVERALL SUMMARY")
print("-" * 80)
print(
    overall.to_string(index=False)
)

print("\nSUMMARY BY CONFORMER COUNT")
print("-" * 80)
print(
    summary.to_string(index=False)
)

print("\n" + "=" * 80)
print("TOP 30 DESCRIPTOR COMPARISONS AFTER FDR")
print("=" * 80)

cols_d = [
    "n_conformers",
    "architecture",
    "descriptor_1",
    "descriptor_2",
    "mean_difference",
    "median_difference",
    "fraction_d1_better",
    "wilcoxon_p",
    "BH_q",
    "significant_fdr_0.05"
]

print(
    descriptor_top[cols_d].to_string(index=False)
)

print("\n" + "=" * 80)
print("TOP 30 ARCHITECTURE COMPARISONS AFTER FDR")
print("=" * 80)

cols_a = [
    "n_conformers",
    "descriptor",
    "architecture_1",
    "architecture_2",
    "mean_difference",
    "median_difference",
    "fraction_a1_better",
    "wilcoxon_p",
    "BH_q",
    "significant_fdr_0.05"
]

print(
    architecture_top[cols_a].to_string(index=False)
)

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)
print(OUT)

print("\nFDR analysis completed successfully.")
