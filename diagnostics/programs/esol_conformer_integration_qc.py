import os
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/Documents/qsarmil")
DATA = os.path.join(BASE, "data")

DIV_FILE = os.path.join(DATA, "esol_conformer_diversity_20conf.csv")

PRED_FILES = {
    5: os.path.join(DATA, "esol_qsarmil_5conf_predictions.csv"),
    10: os.path.join(DATA, "esol_qsarmil_10conf_predictions.csv"),
    15: os.path.join(DATA, "esol_qsarmil_15conf_predictions.csv"),
    20: os.path.join(DATA, "esol_qsarmil_20conf_predictions.csv"),
}

OUTDIR = os.path.join(DATA, "esol_conformer_integration_qc")
os.makedirs(OUTDIR, exist_ok=True)


def normalize_smiles(x):
    return str(x).strip()


print("=" * 80)
print("ESOL CONFORMER / PREDICTION INTEGRATION QC")
print("=" * 80)

# ---------------------------------------------------------------------
# 1. Load diversity data
# ---------------------------------------------------------------------

div = pd.read_csv(DIV_FILE)

print("\nDIVERSITY DATA")
print("-" * 80)
print("Shape:", div.shape)
print("Unique SMILES:", div["smiles"].nunique())
print("Duplicate rows:", div.duplicated().sum())
print("Status counts:")
print(div["status"].value_counts(dropna=False).to_string())

required_div = [
    "smiles",
    "n_requested",
    "n_generated",
    "uff_delta_E_kcal_mol",
    "uff_mean_relative_E_kcal_mol",
    "uff_median_relative_E_kcal_mol",
    "n_low_energy_5kcal",
    "rmsd_mean_A",
    "rmsd_median_A",
    "rmsd_max_A",
    "rmsd_mean_lowE5_A",
    "rmsd_median_lowE5_A",
    "rmsd_max_lowE5_A",
]

missing = [c for c in required_div if c not in div.columns]

if missing:
    raise RuntimeError(
        "Missing diversity columns: " + ", ".join(missing)
    )

# ---------------------------------------------------------------------
# 2. Validate diversity values
# ---------------------------------------------------------------------

print("\nDIVERSITY VALUE QC")
print("-" * 80)

print("n_requested:")
print(div["n_requested"].value_counts().sort_index().to_string())

print("\nn_generated:")
print(div["n_generated"].value_counts().sort_index().to_string())

print("\nNon-finite values:")
for c in required_div[1:]:
    n_bad = (~np.isfinite(pd.to_numeric(div[c], errors="coerce"))).sum()
    if n_bad:
        print(f"  {c}: {n_bad}")

print("\nNegative diversity values:")
for c in [
    "uff_delta_E_kcal_mol",
    "uff_mean_relative_E_kcal_mol",
    "uff_median_relative_E_kcal_mol",
    "rmsd_mean_A",
    "rmsd_median_A",
    "rmsd_max_A",
    "rmsd_mean_lowE5_A",
    "rmsd_median_lowE5_A",
    "rmsd_max_lowE5_A",
]:
    vals = pd.to_numeric(div[c], errors="coerce")
    nneg = (vals < 0).sum()
    print(f"  {c}: {nneg}")

# ---------------------------------------------------------------------
# 3. Load prediction files
# ---------------------------------------------------------------------

pred = {}

for nconf, f in PRED_FILES.items():

    print("\n" + "=" * 80)
    print(f"{nconf}-CONFORMER PREDICTION FILE")
    print("=" * 80)

    if not os.path.exists(f):
        raise FileNotFoundError(f)

    df = pd.read_csv(f)

    pred[nconf] = df

    print("Shape:", df.shape)
    print("Unique SMILES:", df["smiles"].nunique())
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate SMILES:", df["smiles"].duplicated().sum())

    required = [
        "smiles",
        "measured log solubility in mols per litre",
        "RotB",
        "prediction",
        "num_conformers",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{nconf}-conf file missing columns: {missing}"
        )

    print("\nConformer-count values:")
    print(df["num_conformers"].value_counts(dropna=False).to_string())

    print("\nMissing predictions:",
          df["prediction"].isna().sum())

    print("Missing target:",
          df["measured log solubility in mols per litre"].isna().sum())

# ---------------------------------------------------------------------
# 4. Cross-file SMILES validation
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("CROSS-FILE MOLECULE VALIDATION")
print("=" * 80)

reference_smiles = set(div["smiles"].map(normalize_smiles))

for nconf, df in pred.items():

    current = set(df["smiles"].map(normalize_smiles))

    missing_from_prediction = reference_smiles - current
    extra_in_prediction = current - reference_smiles

    print(f"\n{nconf}-conf:")
    print("  diversity molecules:", len(reference_smiles))
    print("  prediction molecules:", len(current))
    print("  missing from prediction:", len(missing_from_prediction))
    print("  extra in prediction:", len(extra_in_prediction))

    if missing_from_prediction:
        print("  Missing examples:",
              list(sorted(missing_from_prediction))[:5])

    if extra_in_prediction:
        print("  Extra examples:",
              list(sorted(extra_in_prediction))[:5])

# ---------------------------------------------------------------------
# 5. Target and RotB consistency
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("TARGET / RotB CONSISTENCY")
print("=" * 80)

target_col = "measured log solubility in mols per litre"

merged = div[["smiles"]].copy()

for nconf, df in pred.items():

    tmp = df[
        [
            "smiles",
            target_col,
            "RotB",
            "prediction",
            "num_conformers",
        ]
    ].copy()

    tmp["smiles"] = tmp["smiles"].map(normalize_smiles)

    tmp = tmp.rename(
        columns={
            target_col: f"target_{nconf}",
            "RotB": f"RotB_{nconf}",
            "prediction": f"prediction_{nconf}",
            "num_conformers": f"nconf_{nconf}",
        }
    )

    merged["smiles"] = merged["smiles"].map(normalize_smiles)

    merged = merged.merge(
        tmp,
        on="smiles",
        how="left",
        validate="one_to_one",
    )

# Compare target and RotB values across prediction files

for nconf in [10, 15, 20]:
    target_diff = (
        merged["target_5"] - merged[f"target_{nconf}"]
    ).abs()

    rotb_diff = (
        merged["RotB_5"] - merged[f"RotB_{nconf}"]
    ).abs()

    print(f"\n5 vs {nconf}:")
    print("  max target difference:",
          target_diff.max())
    print("  max RotB difference:",
          rotb_diff.max())

# ---------------------------------------------------------------------
# 6. Merge diversity variables
# ---------------------------------------------------------------------

div_small = div[
    [
        "smiles",
        "n_generated",
        "uff_E_min_kcal_mol",
        "uff_E_max_kcal_mol",
        "uff_delta_E_kcal_mol",
        "uff_mean_relative_E_kcal_mol",
        "uff_median_relative_E_kcal_mol",
        "n_low_energy_5kcal",
        "rmsd_mean_A",
        "rmsd_median_A",
        "rmsd_max_A",
        "rmsd_mean_lowE5_A",
        "rmsd_median_lowE5_A",
        "rmsd_max_lowE5_A",
    ]
].copy()

div_small["smiles"] = div_small["smiles"].map(normalize_smiles)

merged = merged.merge(
    div_small,
    on="smiles",
    how="left",
    validate="one_to_one",
)

# ---------------------------------------------------------------------
# 7. Construct molecule-level errors
# ---------------------------------------------------------------------

for nconf in [5, 10, 15, 20]:

    merged[f"abs_error_{nconf}"] = (
        merged[f"prediction_{nconf}"]
        - merged["target_5"]
    ).abs()

# Incremental improvement relative to 5-conformer model
# Positive = lower absolute error at the larger conformer count

for nconf in [10, 15, 20]:

    merged[f"improvement_vs_5_{nconf}"] = (
        merged["abs_error_5"]
        - merged[f"abs_error_{nconf}"]
    )

# ---------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("MOLECULE-LEVEL PREDICTION SUMMARY")
print("=" * 80)

for nconf in [5, 10, 15, 20]:

    e = merged[f"abs_error_{nconf}"]

    print(
        f"{nconf:2d} conformers | "
        f"N={e.notna().sum():3d} | "
        f"MAE={e.mean():.6f} | "
        f"median AE={e.median():.6f}"
    )

# ---------------------------------------------------------------------
# 9. Diversity summary
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("CONFORMATIONAL DIVERSITY SUMMARY")
print("=" * 80)

div_cols = [
    "uff_delta_E_kcal_mol",
    "uff_mean_relative_E_kcal_mol",
    "uff_median_relative_E_kcal_mol",
    "n_low_energy_5kcal",
    "rmsd_mean_A",
    "rmsd_median_A",
    "rmsd_max_A",
    "rmsd_mean_lowE5_A",
    "rmsd_median_lowE5_A",
    "rmsd_max_lowE5_A",
]

print(
    merged[div_cols]
    .describe()
    .T
    .to_string()
)

# ---------------------------------------------------------------------
# 10. Identify extreme UFF-energy observations
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("EXTREME UFF ENERGY DISPERSION")
print("=" * 80)

extreme = merged[
    [
        "smiles",
        "uff_delta_E_kcal_mol",
        "uff_mean_relative_E_kcal_mol",
        "n_low_energy_5kcal",
        "rmsd_mean_A",
        "rmsd_max_A",
    ]
].sort_values(
    "uff_delta_E_kcal_mol",
    ascending=False,
)

print(extreme.head(15).to_string(index=False))

# ---------------------------------------------------------------------
# 11. Basic completeness checks
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL COMPLETENESS CHECK")
print("=" * 80)

print("Merged molecules:", len(merged))
print("Unique SMILES:", merged["smiles"].nunique())

for nconf in [5, 10, 15, 20]:
    print(
        f"{nconf}-conf prediction missing:",
        merged[f"prediction_{nconf}"].isna().sum()
    )

print(
    "Diversity missing:",
    merged["uff_delta_E_kcal_mol"].isna().sum()
)

# ---------------------------------------------------------------------
# 12. Save integrated dataset
# ---------------------------------------------------------------------

out_csv = os.path.join(
    OUTDIR,
    "esol_molecule_level_integrated.csv"
)

merged.to_csv(out_csv, index=False)

print("\nSaved:")
print(out_csv)

print("\n" + "=" * 80)
print("QC COMPLETE")
print("=" * 80)
