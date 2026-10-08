import os
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error


BASE = "data"

INPUT = os.path.join(
    BASE,
    "esol_molecule_level_mil_vs_sil",
    "esol_molecule_level_sil_mil_diversity.csv"
)

OUTDIR = os.path.join(
    BASE,
    "esol_error_distribution_analysis"
)

os.makedirs(OUTDIR, exist_ok=True)

TARGET = "measured log solubility in mols per litre"

NCONF_LIST = [5, 10, 15, 20]


# ============================================================
# Load integrated molecule-level dataset
# ============================================================

df = pd.read_csv(INPUT)

print("=" * 90)
print("ESOL ERROR-DISTRIBUTION ANALYSIS")
print("=" * 90)

print(f"Input: {INPUT}")
print(f"Molecules: {len(df)}")


required = [
    "smiles",
    TARGET,
    "SIL_prediction",
    "SIL_abs_error",
]

for nconf in NCONF_LIST:
    required.extend([
        f"MIL_{nconf}",
        f"MIL_{nconf}_abs_error",
        f"improvement_{nconf}",
    ])

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# 1. Top molecules helped / harmed
# ============================================================

for nconf in NCONF_LIST:

    imp_col = f"improvement_{nconf}"
    mil_err_col = f"MIL_{nconf}_abs_error"

    cols = [
        "Compound ID",
        "smiles",
        "RotB",
        "Flexibility",
        TARGET,
        "SIL_prediction",
        "SIL_abs_error",
        f"MIL_{nconf}",
        mil_err_col,
        imp_col,
        "uff_delta_E_kcal_mol",
        "rmsd_mean_A",
        "rmsd_max_A",
        "n_low_energy_5kcal",
    ]

    cols = [c for c in cols if c in df.columns]

    # Largest positive improvement
    helped = (
        df[cols]
        .sort_values(
            imp_col,
            ascending=False
        )
        .head(20)
    )

    helped.to_csv(
        os.path.join(
            OUTDIR,
            f"top20_helped_{nconf}conf.csv"
        ),
        index=False
    )

    # Largest negative improvement
    harmed = (
        df[cols]
        .sort_values(
            imp_col,
            ascending=True
        )
        .head(20)
    )

    harmed.to_csv(
        os.path.join(
            OUTDIR,
            f"top20_harmed_{nconf}conf.csv"
        ),
        index=False
    )


# ============================================================
# 2. Error distribution summary
# ============================================================

summary_rows = []

for nconf in NCONF_LIST:

    sil_err = df["SIL_abs_error"]
    mil_err = df[f"MIL_{nconf}_abs_error"]

    improvement = (
        sil_err - mil_err
    )

    summary_rows.append({
        "n_conformers": nconf,

        "N": len(df),

        "mean_SIL_abs_error":
            sil_err.mean(),

        "mean_MIL_abs_error":
            mil_err.mean(),

        "mean_improvement":
            improvement.mean(),

        "median_improvement":
            improvement.median(),

        "std_improvement":
            improvement.std(ddof=1),

        "min_improvement":
            improvement.min(),

        "max_improvement":
            improvement.max(),

        "Q1_improvement":
            improvement.quantile(0.25),

        "Q3_improvement":
            improvement.quantile(0.75),

        "fraction_improved":
            (improvement > 0).mean(),

        "fraction_worsened":
            (improvement < 0).mean(),

        "fraction_equal":
            (improvement == 0).mean(),

        "mean_absolute_change":
            np.abs(improvement).mean(),
    })


summary = pd.DataFrame(summary_rows)

print("\n" + "=" * 90)
print("ERROR-DISTRIBUTION SUMMARY")
print("=" * 90)

print(summary.to_string(index=False))

summary.to_csv(
    os.path.join(
        OUTDIR,
        "error_distribution_summary.csv"
    ),
    index=False
)


# ============================================================
# 3. Does MIL help molecules with large SIL errors?
# ============================================================

print("\n" + "=" * 90)
print("SIL ERROR MAGNITUDE vs MIL IMPROVEMENT")
print("=" * 90)

corr_rows = []

for nconf in NCONF_LIST:

    x = df["SIL_abs_error"]
    y = df[f"improvement_{nconf}"]

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    rho, p_s = spearmanr(
        x[mask],
        y[mask]
    )

    r, p_p = pearsonr(
        x[mask],
        y[mask]
    )

    corr_rows.append({
        "n_conformers": nconf,
        "N": mask.sum(),
        "Spearman_rho": rho,
        "Spearman_p": p_s,
        "Pearson_r": r,
        "Pearson_p": p_p,
    })


error_corr = pd.DataFrame(corr_rows)

print(
    error_corr.to_string(index=False)
)

error_corr.to_csv(
    os.path.join(
        OUTDIR,
        "sil_error_vs_mil_improvement.csv"
    ),
    index=False
)


# ============================================================
# 4. Stratify molecules by SIL baseline error
# ============================================================

print("\n" + "=" * 90)
print("MIL IMPROVEMENT BY SIL BASELINE ERROR QUANTILE")
print("=" * 90)


df["SIL_error_quantile"] = pd.qcut(
    df["SIL_abs_error"],
    q=4,
    labels=[
        "Q1_lowest_error",
        "Q2",
        "Q3",
        "Q4_highest_error",
    ],
    duplicates="drop",
)


quantile_rows = []

for nconf in NCONF_LIST:

    for group, sub in df.groupby(
        "SIL_error_quantile",
        observed=False
    ):

        imp = sub[f"improvement_{nconf}"]

        quantile_rows.append({
            "n_conformers": nconf,
            "SIL_error_quantile": str(group),
            "N": len(sub),

            "mean_SIL_abs_error":
                sub["SIL_abs_error"].mean(),

            "mean_MIL_abs_error":
                sub[
                    f"MIL_{nconf}_abs_error"
                ].mean(),

            "mean_improvement":
                imp.mean(),

            "median_improvement":
                imp.median(),

            "fraction_improved":
                (imp > 0).mean(),
        })


quantiles = pd.DataFrame(
    quantile_rows
)

print(
    quantiles.to_string(index=False)
)

quantiles.to_csv(
    os.path.join(
        OUTDIR,
        "improvement_by_sil_error_quantile.csv"
    ),
    index=False
)


# ============================================================
# 5. Worst-SIL molecules: top 5%, 10%, 20%
# ============================================================

print("\n" + "=" * 90)
print("PERFORMANCE ON MOLECULES WITH LARGEST SIL ERRORS")
print("=" * 90)


worst_rows = []

for fraction in [0.05, 0.10, 0.20]:

    threshold = df["SIL_abs_error"].quantile(
        1.0 - fraction
    )

    sub = df[
        df["SIL_abs_error"] >= threshold
    ]

    print(
        f"\nWorst {fraction*100:.0f}%: "
        f"N={len(sub)}, "
        f"SIL error threshold={threshold:.6f}"
    )

    for nconf in NCONF_LIST:

        sil_rmse = np.sqrt(
            np.mean(
                sub["SIL_abs_error"] ** 2
            )
        )

        mil_rmse = np.sqrt(
            np.mean(
                sub[f"MIL_{nconf}_abs_error"] ** 2
            )
        )

        sil_mae = (
            sub["SIL_abs_error"].mean()
        )

        mil_mae = (
            sub[f"MIL_{nconf}_abs_error"].mean()
        )

        imp = (
            sub["SIL_abs_error"]
            - sub[f"MIL_{nconf}_abs_error"]
        )

        row = {
            "worst_fraction": fraction,
            "N": len(sub),
            "SIL_error_threshold": threshold,
            "n_conformers": nconf,

            "SIL_RMSE":
                sil_rmse,

            "MIL_RMSE":
                mil_rmse,

            "RMSE_improvement":
                sil_rmse - mil_rmse,

            "SIL_MAE":
                sil_mae,

            "MIL_MAE":
                mil_mae,

            "MAE_improvement":
                sil_mae - mil_mae,

            "fraction_improved":
                (imp > 0).mean(),

            "mean_improvement":
                imp.mean(),
        }

        worst_rows.append(row)


worst = pd.DataFrame(worst_rows)

print(
    worst.to_string(index=False)
)

worst.to_csv(
    os.path.join(
        OUTDIR,
        "worst_sil_error_analysis.csv"
    ),
    index=False
)


# ============================================================
# 6. Does flexibility modify the relationship?
# ============================================================

print("\n" + "=" * 90)
print("SIL ERROR vs MIL IMPROVEMENT WITHIN FLEXIBILITY GROUPS")
print("=" * 90)


flex_corr_rows = []

for nconf in NCONF_LIST:

    for group, sub in df.groupby(
        "Flexibility",
        observed=False
    ):

        x = sub["SIL_abs_error"]
        y = sub[f"improvement_{nconf}"]

        mask = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        if mask.sum() < 8:
            continue

        rho, p = spearmanr(
            x[mask],
            y[mask]
        )

        flex_corr_rows.append({
            "n_conformers": nconf,
            "Flexibility": str(group),
            "N": mask.sum(),
            "Spearman_rho": rho,
            "Spearman_p": p,
        })


flex_corr = pd.DataFrame(
    flex_corr_rows
)

print(
    flex_corr.to_string(index=False)
)

flex_corr.to_csv(
    os.path.join(
        OUTDIR,
        "sil_error_vs_improvement_by_flexibility.csv"
    ),
    index=False
)


# ============================================================
# 7. Largest individual squared-error corrections
# ============================================================

print("\n" + "=" * 90)
print("LARGEST SQUARED-ERROR CORRECTIONS")
print("=" * 90)


sq_rows = []

for nconf in NCONF_LIST:

    sil_sq = (
        df["SIL_prediction"] - df[TARGET]
    ) ** 2

    mil_sq = (
        df[f"MIL_{nconf}"] - df[TARGET]
    ) ** 2

    correction = (
        sil_sq - mil_sq
    )

    tmp = df[
        [
            "Compound ID",
            "smiles",
            "RotB",
            "Flexibility",
            TARGET,
            "SIL_prediction",
            f"MIL_{nconf}",
            "SIL_abs_error",
            f"MIL_{nconf}_abs_error",
            "uff_delta_E_kcal_mol",
            "rmsd_mean_A",
            "rmsd_max_A",
        ]
    ].copy()

    tmp["squared_error_correction"] = correction

    tmp = tmp.sort_values(
        "squared_error_correction",
        ascending=False
    )

    tmp.head(20).to_csv(
        os.path.join(
            OUTDIR,
            f"top20_squared_error_corrections_{nconf}conf.csv"
        ),
        index=False
    )

    sq_rows.append({
        "n_conformers": nconf,
        "largest_squared_error_correction":
            correction.max(),
        "smallest_squared_error_correction":
            correction.min(),
        "sum_squared_error_correction":
            correction.sum(),
    })


sq_summary = pd.DataFrame(sq_rows)

print(
    sq_summary.to_string(index=False)
)

sq_summary.to_csv(
    os.path.join(
        OUTDIR,
        "squared_error_correction_summary.csv"
    ),
    index=False
)


# ============================================================
# 8. Compare RMSE contribution of worst errors
# ============================================================

print("\n" + "=" * 90)
print("RMSE CONTRIBUTION ANALYSIS")
print("=" * 90)


contribution_rows = []

for nconf in NCONF_LIST:

    sil_sq = (
        df["SIL_prediction"] - df[TARGET]
    ) ** 2

    mil_sq = (
        df[f"MIL_{nconf}"] - df[TARGET]
    ) ** 2

    tmp = pd.DataFrame({
        "SIL_sq": sil_sq,
        "MIL_sq": mil_sq,
    })

    # Rank by SIL error
    tmp["rank"] = (
        tmp["SIL_sq"]
        .rank(
            method="first",
            ascending=False
        )
    )

    for top_fraction in [
        0.05,
        0.10,
        0.20,
        0.50,
        1.00,
    ]:

        n = int(
            np.ceil(
                len(tmp) * top_fraction
            )
        )

        sub = tmp.nsmallest(
            n,
            "rank"
        )

        contribution_rows.append({
            "n_conformers": nconf,
            "fraction_of_molecules":
                top_fraction,
            "N": len(sub),

            "SIL_sum_squared_error":
                sub["SIL_sq"].sum(),

            "MIL_sum_squared_error":
                sub["MIL_sq"].sum(),

            "squared_error_reduction":
                (
                    sub["SIL_sq"].sum()
                    - sub["MIL_sq"].sum()
                ),

            "fraction_of_total_SIL_SE":
                (
                    sub["SIL_sq"].sum()
                    / tmp["SIL_sq"].sum()
                ),

            "fraction_of_total_MIL_SE":
                (
                    sub["MIL_sq"].sum()
                    / tmp["MIL_sq"].sum()
                ),
        })


contribution = pd.DataFrame(
    contribution_rows
)

print(
    contribution.to_string(index=False)
)

contribution.to_csv(
    os.path.join(
        OUTDIR,
        "rmse_contribution_by_sil_error_rank.csv"
    ),
    index=False
)


# ============================================================
# Save augmented dataset
# ============================================================

df.to_csv(
    os.path.join(
        OUTDIR,
        "esol_error_distribution_augmented.csv"
    ),
    index=False
)


print("\n" + "=" * 90)
print("OUTPUT FILES")
print("=" * 90)

print(
    f"""
{OUTDIR}/

    error_distribution_summary.csv
    sil_error_vs_mil_improvement.csv
    improvement_by_sil_error_quantile.csv
    worst_sil_error_analysis.csv
    sil_error_vs_improvement_by_flexibility.csv
    squared_error_correction_summary.csv
    rmse_contribution_by_sil_error_rank.csv
    esol_error_distribution_augmented.csv

    top20_helped_*conf.csv
    top20_harmed_*conf.csv
    top20_squared_error_corrections_*conf.csv
"""
)

print("\nAnalysis completed successfully.")
