import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import wilcoxon


BASE = "data"

SIL_FILE = os.path.join(BASE, "esol_sil_predictions.csv")

MIL_FILES = {
    5: os.path.join(BASE, "esol_qsarmil_5conf_predictions.csv"),
    10: os.path.join(BASE, "esol_qsarmil_10conf_predictions.csv"),
    15: os.path.join(BASE, "esol_qsarmil_15conf_predictions.csv"),
    20: os.path.join(BASE, "esol_qsarmil_20conf_predictions.csv"),
}

DIVERSITY_FILE = os.path.join(
    BASE,
    "esol_conformer_diversity_20conf.csv"
)

OUTDIR = os.path.join(
    BASE,
    "esol_molecule_level_mil_vs_sil"
)

os.makedirs(OUTDIR, exist_ok=True)


TARGET = "measured log solubility in mols per litre"


# ============================================================
# Load SIL
# ============================================================

sil = pd.read_csv(SIL_FILE)

required_sil = [
    "Compound ID",
    "smiles",
    TARGET,
    "RotB",
    "Flexibility",
    "prediction",
]

missing = [c for c in required_sil if c not in sil.columns]

if missing:
    raise ValueError(
        f"Missing SIL columns: {missing}"
    )

sil = sil.rename(
    columns={"prediction": "SIL_prediction"}
)

print("=" * 80)
print("MOLECULE-LEVEL SIL vs MIL ANALYSIS")
print("=" * 80)

print(f"SIL molecules: {len(sil)}")


# ============================================================
# Validate SIL uniqueness
# ============================================================

if sil["smiles"].duplicated().any():
    raise ValueError(
        "Duplicate SMILES found in SIL predictions."
    )

if sil["Compound ID"].duplicated().any():
    raise ValueError(
        "Duplicate Compound IDs found in SIL predictions."
    )


# ============================================================
# Load diversity
# ============================================================

div = pd.read_csv(DIVERSITY_FILE)

print(f"Diversity molecules: {len(div)}")

if "smiles" not in div.columns:
    raise ValueError(
        "Diversity file does not contain 'smiles'."
    )


if div["smiles"].duplicated().any():
    raise ValueError(
        "Duplicate SMILES found in diversity file."
    )


# ============================================================
# Merge diversity onto SIL
# ============================================================

df = sil.copy()

df = df.merge(
    div,
    on="smiles",
    how="left",
    suffixes=("", "_div"),
    validate="one_to_one",
)

if len(df) != len(sil):
    raise ValueError(
        "Unexpected row count after SIL/diversity merge."
    )

if df["n_generated"].isna().any():
    raise ValueError(
        "Some SIL molecules are missing diversity data."
    )


# ============================================================
# Load MIL predictions
# ============================================================

for nconf, filepath in MIL_FILES.items():

    mil = pd.read_csv(filepath)

    print(
        f"{nconf:2d}-conf MIL molecules: {len(mil)}"
    )

    required = [
        "smiles",
        TARGET,
        "prediction",
        "num_conformers",
    ]

    missing = [c for c in required if c not in mil.columns]

    if missing:
        raise ValueError(
            f"{nconf}-conf file missing columns: {missing}"
        )

    if mil["smiles"].duplicated().any():
        raise ValueError(
            f"Duplicate SMILES in {nconf}-conf MIL file."
        )

    # Rename prediction
    mil = mil[
        [
            "smiles",
            TARGET,
            "prediction",
            "num_conformers",
        ]
    ].rename(
        columns={
            TARGET: f"target_{nconf}",
            "prediction": f"MIL_{nconf}",
            "num_conformers": f"nconf_{nconf}",
        }
    )

    df = df.merge(
        mil,
        on="smiles",
        how="left",
        validate="one_to_one",
    )

    if df[f"MIL_{nconf}"].isna().any():
        raise ValueError(
            f"Missing MIL predictions for {nconf} conf."
        )


# ============================================================
# Cross-check targets
# ============================================================

for nconf in MIL_FILES:

    diff = np.abs(
        df[TARGET].values
        - df[f"target_{nconf}"].values
    )

    print(
        f"Target difference SIL vs {nconf}-conf: "
        f"max = {diff.max():.12g}"
    )

    if not np.allclose(diff, 0):
        raise ValueError(
            f"Target mismatch for {nconf}-conf."
        )


# ============================================================
# Calculate molecule-level errors
# ============================================================

df["SIL_error"] = (
    df["SIL_prediction"] - df[TARGET]
)

df["SIL_abs_error"] = np.abs(
    df["SIL_error"]
)


for nconf in MIL_FILES:

    df[f"MIL_{nconf}_error"] = (
        df[f"MIL_{nconf}"] - df[TARGET]
    )

    df[f"MIL_{nconf}_abs_error"] = np.abs(
        df[f"MIL_{nconf}_error"]
    )

    # Positive = MIL is better
    df[f"improvement_{nconf}"] = (
        df["SIL_abs_error"]
        - df[f"MIL_{nconf}_abs_error"]
    )

    # Positive = MIL is better
    df[f"squared_error_improvement_{nconf}"] = (
        df["SIL_error"] ** 2
        - df[f"MIL_{nconf}_error"] ** 2
    )


# ============================================================
# Overall performance
# ============================================================

print("\n" + "=" * 80)
print("OVERALL PERFORMANCE")
print("=" * 80)

rows = []

for nconf in MIL_FILES:

    y = df[TARGET]

    sil_pred = df["SIL_prediction"]
    mil_pred = df[f"MIL_{nconf}"]

    sil_rmse = np.sqrt(
        mean_squared_error(y, sil_pred)
    )

    mil_rmse = np.sqrt(
        mean_squared_error(y, mil_pred)
    )

    sil_mae = mean_absolute_error(
        y, sil_pred
    )

    mil_mae = mean_absolute_error(
        y, mil_pred
    )

    rows.append({
        "n_conformers": nconf,

        "SIL_R2": r2_score(y, sil_pred),
        "MIL_R2": r2_score(y, mil_pred),

        "SIL_RMSE": sil_rmse,
        "MIL_RMSE": mil_rmse,

        "RMSE_improvement": (
            sil_rmse - mil_rmse
        ),

        "SIL_MAE": sil_mae,
        "MIL_MAE": mil_mae,

        "MAE_improvement": (
            sil_mae - mil_mae
        ),

        "SIL_Pearson_r": np.corrcoef(
            y, sil_pred
        )[0, 1],

        "MIL_Pearson_r": np.corrcoef(
            y, mil_pred
        )[0, 1],
    })


overall = pd.DataFrame(rows)

print(
    overall.to_string(index=False)
)

overall.to_csv(
    os.path.join(
        OUTDIR,
        "overall_sil_vs_mil.csv"
    ),
    index=False
)


# ============================================================
# Molecule-level improvement summary
# ============================================================

summary_rows = []

for nconf in MIL_FILES:

    imp = df[f"improvement_{nconf}"]

    improved = (imp > 0).sum()
    worsened = (imp < 0).sum()
    equal = (imp == 0).sum()

    summary_rows.append({
        "n_conformers": nconf,
        "N": len(df),
        "improved": improved,
        "worsened": worsened,
        "equal": equal,
        "fraction_improved": improved / len(df),
        "mean_improvement_MAE": imp.mean(),
        "median_improvement_MAE": imp.median(),
        "mean_absolute_change": np.abs(imp).mean(),
    })


molecule_summary = pd.DataFrame(
    summary_rows
)

print("\n" + "=" * 80)
print("MOLECULE-LEVEL IMPROVEMENT")
print("=" * 80)

print(
    molecule_summary.to_string(index=False)
)

molecule_summary.to_csv(
    os.path.join(
        OUTDIR,
        "molecule_level_improvement_summary.csv"
    ),
    index=False
)


# ============================================================
# Flexibility analysis
# ============================================================

print("\n" + "=" * 80)
print("IMPROVEMENT BY FLEXIBILITY")
print("=" * 80)

flex_rows = []

for nconf in MIL_FILES:

    for group, sub in df.groupby(
        "Flexibility",
        observed=False
    ):

        imp = sub[f"improvement_{nconf}"]

        flex_rows.append({
            "n_conformers": nconf,
            "Flexibility": str(group),
            "N": len(sub),
            "mean_improvement": imp.mean(),
            "median_improvement": imp.median(),
            "fraction_improved": (
                (imp > 0).mean()
            ),
            "mean_SIL_abs_error": (
                sub["SIL_abs_error"].mean()
            ),
            "mean_MIL_abs_error": (
                sub[f"MIL_{nconf}_abs_error"].mean()
            ),
        })


flex = pd.DataFrame(flex_rows)

print(
    flex.to_string(index=False)
)

flex.to_csv(
    os.path.join(
        OUTDIR,
        "improvement_by_flexibility.csv"
    ),
    index=False
)


# ============================================================
# Diversity correlations
# ============================================================

print("\n" + "=" * 80)
print("CORRELATION: CONFORMER DIVERSITY vs MIL IMPROVEMENT")
print("=" * 80)


diversity_variables = [
    "UFF_deltaE_kcal",
    "UFF_mean_relE_kcal",
    "UFF_median_relE_kcal",
    "n_low_energy_5kcal",
    "rmsd_mean_all_A",
    "rmsd_median_all_A",
    "rmsd_max_all_A",
    "rmsd_mean_lowE5_A",
    "rmsd_median_lowE5_A",
    "rmsd_max_lowE5_A",
]


corr_rows = []


for nconf in MIL_FILES:

    imp = df[f"improvement_{nconf}"]

    for var in diversity_variables:

        if var not in df.columns:
            continue

        x = df[var]

        mask = (
            np.isfinite(x)
            & np.isfinite(imp)
        )

        if mask.sum() < 10:
            continue

        rho, p = spearmanr(
            x[mask],
            imp[mask]
        )

        r, pp = pearsonr(
            x[mask],
            imp[mask]
        )

        corr_rows.append({
            "n_conformers": nconf,
            "diversity_variable": var,
            "N": mask.sum(),
            "Spearman_rho": rho,
            "Spearman_p": p,
            "Pearson_r": r,
            "Pearson_p": pp,
        })


corr = pd.DataFrame(corr_rows)

print(
    corr.to_string(index=False)
)

corr.to_csv(
    os.path.join(
        OUTDIR,
        "diversity_vs_improvement_correlations.csv"
    ),
    index=False
)


# ============================================================
# Wilcoxon: SIL vs MIL absolute errors
# ============================================================

print("\n" + "=" * 80)
print("PAIRED SIL vs MIL ERROR TESTS")
print("=" * 80)


test_rows = []

for nconf in MIL_FILES:

    sil_abs = df["SIL_abs_error"]
    mil_abs = df[f"MIL_{nconf}_abs_error"]

    result = wilcoxon(
        sil_abs,
        mil_abs,
        alternative="two-sided"
    )

    test_rows.append({
        "n_conformers": nconf,
        "N": len(df),
        "Wilcoxon_statistic": result.statistic,
        "Wilcoxon_p": result.pvalue,
        "mean_SIL_abs_error": sil_abs.mean(),
        "mean_MIL_abs_error": mil_abs.mean(),
    })


tests = pd.DataFrame(test_rows)

print(
    tests.to_string(index=False)
)

tests.to_csv(
    os.path.join(
        OUTDIR,
        "paired_sil_mil_tests.csv"
    ),
    index=False
)


# ============================================================
# Save complete integrated table
# ============================================================

outfile = os.path.join(
    OUTDIR,
    "esol_molecule_level_sil_mil_diversity.csv"
)

df.to_csv(
    outfile,
    index=False
)

print("\n" + "=" * 80)
print("FILES WRITTEN")
print("=" * 80)

print(
    f"""
{OUTDIR}/
    overall_sil_vs_mil.csv
    molecule_level_improvement_summary.csv
    improvement_by_flexibility.csv
    diversity_vs_improvement_correlations.csv
    paired_sil_mil_tests.csv
    esol_molecule_level_sil_mil_diversity.csv
"""
)

print("\nAnalysis completed successfully.")
