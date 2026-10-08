import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


BASE = "data"
SIL_FILE = f"{BASE}/esol_3d_sil_neural/test_predictions.csv"
SIL_METRICS = f"{BASE}/esol_3d_sil_neural/model_metrics.csv"

MIL_DIR = f"{BASE}/esol_72model_audit"

OUTDIR = f"{BASE}/esol_3d_sil_vs_3d_mil"
os.makedirs(OUTDIR, exist_ok=True)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r = pearsonr(y_true, y_pred)[0]

    return r2, rmse, mae, r


def find_smiles_column(df):
    for c in ["SMILES", "smiles", "Smiles"]:
        if c in df.columns:
            return c
    raise RuntimeError("No SMILES column found")


def find_target_column(df):
    candidates = [
        "Y_TRUE_ESOL",
        "measured log solubility in mols per litre",
        "measured log solubility",
        "target",
        "y_true",
        "Y_TRUE",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise RuntimeError(
        "Could not identify target column. Available columns:\n"
        + "\n".join(map(str, df.columns))
    )


print("=" * 80)
print("3D-SIL vs 3D-MIL MATCHED MODEL COMPARISON")
print("=" * 80)


# -------------------------------------------------------------------------
# 1. Read 3D-SIL results
# -------------------------------------------------------------------------

sil = pd.read_csv(SIL_FILE)

sil_smiles_col = find_smiles_column(sil)
sil_target_col = find_target_column(sil)

print(f"\n3D-SIL file: {SIL_FILE}")
print(f"Rows: {len(sil)}")
print(f"SMILES column: {sil_smiles_col}")
print(f"Target column: {sil_target_col}")


if sil[sil_smiles_col].duplicated().any():
    dup = sil.loc[
        sil[sil_smiles_col].duplicated(keep=False),
        sil_smiles_col
    ].tolist()

    raise RuntimeError(
        f"Duplicate SMILES found in 3D-SIL predictions: {dup[:10]}"
    )


# -------------------------------------------------------------------------
# 2. Identify successful 3D-SIL model columns
# -------------------------------------------------------------------------

non_model_cols = {
    sil_smiles_col,
    sil_target_col,
    "Compound ID",
    "compound_id",
    "SMILES",
    "smiles",
}

sil_model_cols = []

for c in sil.columns:
    if c in non_model_cols:
        continue

    # Only keep numeric prediction columns
    if pd.api.types.is_numeric_dtype(sil[c]):
        sil_model_cols.append(c)


print(f"Successful 3D-SIL model columns detected: {len(sil_model_cols)}")


# -------------------------------------------------------------------------
# 3. Read 3D-SIL model metrics if available
# -------------------------------------------------------------------------

if os.path.exists(SIL_METRICS):
    sil_metrics = pd.read_csv(SIL_METRICS)
    print(f"3D-SIL model metrics file: {SIL_METRICS}")
else:
    sil_metrics = None
    print("WARNING: 3D-SIL model_metrics.csv not found.")


# -------------------------------------------------------------------------
# 4. Process each conformer-count MIL file
# -------------------------------------------------------------------------

all_results = []

for nconf in [5, 10, 15, 20]:

    mil_file = (
        f"{MIL_DIR}/matched_test_predictions_{nconf}conf.csv"
    )

    print("\n" + "-" * 80)
    print(f"Processing {nconf}-conformer MIL")
    print(f"File: {mil_file}")

    mil = pd.read_csv(mil_file)

    mil_smiles_col = find_smiles_column(mil)

    print(f"MIL rows: {len(mil)}")
    print(f"MIL SMILES column: {mil_smiles_col}")

    # Check duplicates
    if mil[mil_smiles_col].duplicated().any():
        dup = mil.loc[
            mil[mil_smiles_col].duplicated(keep=False),
            mil_smiles_col
        ].tolist()

        raise RuntimeError(
            f"Duplicate SMILES found in {nconf}-conf MIL file: {dup[:10]}"
        )

    # ---------------------------------------------------------------------
    # Match molecules by SMILES rather than row order
    # ---------------------------------------------------------------------

    sil_keys = set(sil[sil_smiles_col].astype(str))
    mil_keys = set(mil[mil_smiles_col].astype(str))

    only_sil = sil_keys - mil_keys
    only_mil = mil_keys - sil_keys

    print(f"Common molecules: {len(sil_keys & mil_keys)}")
    print(f"Only in 3D-SIL: {len(only_sil)}")
    print(f"Only in 3D-MIL: {len(only_mil)}")

    if only_sil or only_mil:
        raise RuntimeError(
            f"Molecule-set mismatch for {nconf}-conf MIL. "
            f"Only SIL={len(only_sil)}, only MIL={len(only_mil)}"
        )

    # Normalize SMILES to common column
    sil_tmp = sil.copy()
    mil_tmp = mil.copy()

    sil_tmp["__SMILES_KEY__"] = sil_tmp[sil_smiles_col].astype(str)
    mil_tmp["__SMILES_KEY__"] = mil_tmp[mil_smiles_col].astype(str)

    # ---------------------------------------------------------------------
    # Determine MIL model columns
    # ---------------------------------------------------------------------

    mil_non_model = {
        mil_smiles_col,
        "SMILES",
        "smiles",
        "Compound ID",
        "compound_id",
        "Y_TRUE_ESOL",
        "Y_TRUE",
        "y_true",
        "measured log solubility",
        "__SMILES_KEY__",
    }

    mil_model_cols = []

    for c in mil.columns:
        if c in mil_non_model:
            continue

        if pd.api.types.is_numeric_dtype(mil[c]):
            mil_model_cols.append(c)

    print(f"MIL numeric model columns: {len(mil_model_cols)}")

    # ---------------------------------------------------------------------
    # Find common model names
    # ---------------------------------------------------------------------

    common_models = [
        c for c in sil_model_cols
        if c in mil_model_cols
    ]

    print(f"Matched model names: {len(common_models)}")

    if len(common_models) == 0:
        print("\n3D-SIL model names:")
        for c in sil_model_cols:
            print("  ", c)

        print("\n3D-MIL model names:")
        for c in mil_model_cols:
            print("  ", c)

        raise RuntimeError(
            "No common model names found between 3D-SIL and 3D-MIL."
        )

    # ---------------------------------------------------------------------
    # Merge by SMILES
    # ---------------------------------------------------------------------

    sil_keep = [
        "__SMILES_KEY__",
        sil_target_col,
    ] + common_models

    mil_keep = [
        "__SMILES_KEY__",
    ] + common_models

    sil_m = sil_tmp[sil_keep].copy()
    mil_m = mil_tmp[mil_keep].copy()

    # Rename SIL target
    sil_m = sil_m.rename(
        columns={sil_target_col: "Y_TRUE_ESOL"}
    )

    merged = pd.merge(
        sil_m,
        mil_m,
        on="__SMILES_KEY__",
        how="inner",
        suffixes=("_SIL", "_MIL"),
        validate="one_to_one",
    )

    print(f"Merged molecules: {len(merged)}")

    if len(merged) != len(sil):
        raise RuntimeError(
            f"Expected {len(sil)} merged molecules, got {len(merged)}"
        )

    # ---------------------------------------------------------------------
    # Target consistency
    # ---------------------------------------------------------------------

    y_true = merged["Y_TRUE_ESOL"].astype(float).to_numpy()

    # ---------------------------------------------------------------------
    # Model-by-model comparison
    # ---------------------------------------------------------------------

    for model in common_models:

        sil_col = f"{model}_SIL"
        mil_col = f"{model}_MIL"

        y_sil = merged[sil_col].astype(float).to_numpy()
        y_mil = merged[mil_col].astype(float).to_numpy()

        if not np.isfinite(y_sil).all():
            raise RuntimeError(
                f"Non-finite values in 3D-SIL model: {model}"
            )

        if not np.isfinite(y_mil).all():
            raise RuntimeError(
                f"Non-finite values in {nconf}-conf MIL model: {model}"
            )

        sil_r2, sil_rmse, sil_mae, sil_r = metrics(
            y_true, y_sil
        )

        mil_r2, mil_rmse, mil_mae, mil_r = metrics(
            y_true, y_mil
        )

        delta_r2 = mil_r2 - sil_r2

        # Positive = MIL is better
        delta_rmse = sil_rmse - mil_rmse
        delta_mae = sil_mae - mil_mae

        all_results.append({
            "n_conformers": nconf,
            "model": model,

            "SIL_R2": sil_r2,
            "SIL_RMSE": sil_rmse,
            "SIL_MAE": sil_mae,
            "SIL_Pearson_r": sil_r,

            "MIL_R2": mil_r2,
            "MIL_RMSE": mil_rmse,
            "MIL_MAE": mil_mae,
            "MIL_Pearson_r": mil_r,

            "Delta_R2_MIL_minus_SIL": delta_r2,

            # Positive = MIL improves over SIL
            "Delta_RMSE_SIL_minus_MIL": delta_rmse,
            "Delta_MAE_SIL_minus_MIL": delta_mae,

            "MIL_improves_RMSE": delta_rmse > 0,
            "MIL_improves_MAE": delta_mae > 0,
        })


# -------------------------------------------------------------------------
# 5. Save detailed comparison
# -------------------------------------------------------------------------

results = pd.DataFrame(all_results)

detail_file = f"{OUTDIR}/paired_model_comparison.csv"
results.to_csv(detail_file, index=False)

print("\n" + "=" * 80)
print("MATCHED MODEL COMPARISON COMPLETE")
print("=" * 80)

print(f"\nTotal matched model records: {len(results)}")
print(f"Detailed output: {detail_file}")


# -------------------------------------------------------------------------
# 6. Summary by conformer count
# -------------------------------------------------------------------------

summary = (
    results
    .groupby("n_conformers")
    .agg(
        n_models=("model", "count"),
        mean_SIL_RMSE=("SIL_RMSE", "mean"),
        mean_MIL_RMSE=("MIL_RMSE", "mean"),
        mean_Delta_RMSE=("Delta_RMSE_SIL_minus_MIL", "mean"),
        median_Delta_RMSE=("Delta_RMSE_SIL_minus_MIL", "median"),
        mean_Delta_MAE=("Delta_MAE_SIL_minus_MIL", "mean"),
        median_Delta_MAE=("Delta_MAE_SIL_minus_MIL", "median"),
        models_improved_RMSE=("MIL_improves_RMSE", "sum"),
        models_improved_MAE=("MIL_improves_MAE", "sum"),
    )
    .reset_index()
)

summary["fraction_models_improved_RMSE"] = (
    summary["models_improved_RMSE"] / summary["n_models"]
)

summary["fraction_models_improved_MAE"] = (
    summary["models_improved_MAE"] / summary["n_models"]
)

summary_file = f"{OUTDIR}/paired_summary_by_conformer_count.csv"
summary.to_csv(summary_file, index=False)

print("\nSUMMARY BY CONFORMER COUNT")
print(summary.to_string(index=False))


# -------------------------------------------------------------------------
# 7. Descriptor summary
# -------------------------------------------------------------------------

def descriptor_from_model(name):
    parts = name.split("_", 1)
    return parts[0] if parts else name


results["descriptor"] = results["model"].map(descriptor_from_model)

descriptor_summary = (
    results
    .groupby(["n_conformers", "descriptor"])
    .agg(
        n_models=("model", "count"),
        mean_Delta_RMSE=("Delta_RMSE_SIL_minus_MIL", "mean"),
        median_Delta_RMSE=("Delta_RMSE_SIL_minus_MIL", "median"),
        mean_Delta_MAE=("Delta_MAE_SIL_minus_MIL", "mean"),
        models_improved=("MIL_improves_RMSE", "sum"),
    )
    .reset_index()
)

descriptor_file = f"{OUTDIR}/paired_descriptor_summary.csv"
descriptor_summary.to_csv(descriptor_file, index=False)


# -------------------------------------------------------------------------
# 8. Architecture summary
# -------------------------------------------------------------------------

architecture_names = [
    "MeanInstanceWrapperMLP",
    "MeanBagWrapperMLP",
    "MeanInstanceNetworkRegressor",
    "MeanBagNetworkRegressor",
    "AdditiveAttentionNetworkRegressor",
    "HopfieldNetworkRegressor",
    "SelfAttentionNetworkRegressor",
    "DynamicPoolingNetworkRegressor",
]


def architecture_from_model(name):
    for arch in architecture_names:
        if name.endswith(arch):
            return arch
    return "UNKNOWN"


results["architecture"] = results["model"].map(
    architecture_from_model
)

architecture_summary = (
    results
    .groupby(["n_conformers", "architecture"])
    .agg(
        n_models=("model", "count"),
        mean_Delta_RMSE=("Delta_RMSE_SIL_minus_MIL", "mean"),
        median_Delta_RMSE=("Delta_RMSE_SIL_minus_MIL", "median"),
        mean_Delta_MAE=("Delta_MAE_SIL_minus_MIL", "mean"),
        models_improved=("MIL_improves_RMSE", "sum"),
    )
    .reset_index()
)

architecture_file = f"{OUTDIR}/paired_architecture_summary.csv"
architecture_summary.to_csv(
    architecture_file,
    index=False
)


# -------------------------------------------------------------------------
# 9. Top 20 models at 20 conformers
# -------------------------------------------------------------------------

top20 = (
    results[results["n_conformers"] == 20]
    .sort_values(
        "Delta_RMSE_SIL_minus_MIL",
        ascending=False
    )
    .head(20)
)

print("\n" + "=" * 80)
print("TOP 20 MODELS: 20-CONFORMER MIL IMPROVEMENT OVER 3D-SIL")
print("=" * 80)

print(
    top20[
        [
            "model",
            "SIL_RMSE",
            "MIL_RMSE",
            "Delta_RMSE_SIL_minus_MIL",
            "SIL_MAE",
            "MIL_MAE",
            "Delta_MAE_SIL_minus_MIL",
        ]
    ].to_string(index=False)
)


# -------------------------------------------------------------------------
# 10. Largest deteriorations at 20 conformers
# -------------------------------------------------------------------------

bottom20 = (
    results[results["n_conformers"] == 20]
    .sort_values(
        "Delta_RMSE_SIL_minus_MIL",
        ascending=True
    )
    .head(20)
)

print("\n" + "=" * 80)
print("20-CONFORMER MODELS WITH LARGEST DETERIORATION")
print("=" * 80)

print(
    bottom20[
        [
            "model",
            "SIL_RMSE",
            "MIL_RMSE",
            "Delta_RMSE_SIL_minus_MIL",
            "SIL_MAE",
            "MIL_MAE",
            "Delta_MAE_SIL_minus_MIL",
        ]
    ].to_string(index=False)
)


print("\nOutput files:")
print(" ", detail_file)
print(" ", summary_file)
print(" ", descriptor_file)
print(" ", architecture_file)

print("\nDONE")
