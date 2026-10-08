from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path.home() / "Documents" / "qsarmil"
IN_DIR = ROOT / "freesolv_qsarmil_5conf"
OUT_DIR = ROOT / "data" / "freesolv_descriptor_architecture_interaction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = pd.read_csv(IN_DIR / "train.csv")
VAL = pd.read_csv(IN_DIR / "val.csv")
TEST = pd.read_csv(IN_DIR / "test.csv")

# QSARmil output has one Y_TRUE column plus 72 prediction columns.
target_cols = [c for c in TRAIN.columns if c.upper().startswith("Y_TRUE")]
if len(target_cols) != 1:
    raise RuntimeError(f"Expected exactly one Y_TRUE column in train.csv, found {target_cols}")
target = target_cols[0]

# Keep only prediction columns common to all three files.
# Keep only columns that correspond exactly to the 9 x 8 QSARmil models.
descriptors = [
    "RDKitGEOM", "RDKitAUTOCORR", "RDKitRDF", "RDKitMORSE", "RDKitWHIM",
    "MolFeatUSRD", "MolFeatElectroShape", "RDKitGETAWAY", "MolFeatPmapper"
]

architectures = [
    "MeanInstanceWrapperMLPNetworkRegressor",
    "MeanBagWrapperMLPNetworkRegressor",
    "MeanBagNetworkRegressor",
    "MeanInstanceNetworkRegressor",
    "AdditiveAttentionNetworkRegressor",
    "SelfAttentionNetworkRegressor",
    "HopfieldAttentionNetworkRegressor",
    "DynamicPoolingNetworkRegressor"
]

expected_models = [
    d + "|" + a
    for d in descriptors
    for a in architectures
]

pred_cols = [
    c for c in expected_models
    if c in TRAIN.columns and c in VAL.columns and c in TEST.columns
]

missing_models = [c for c in expected_models if c not in pred_cols]

if missing_models:
    raise RuntimeError(
        "Missing expected QSARmil model columns:\n" +
        "\n".join(missing_models)
    )

if len(pred_cols) != 72:
    raise RuntimeError(
        f"Expected 72 QSARmil model prediction columns, found {len(pred_cols)}"
    )

print(f"Verified exactly {len(pred_cols)} QSARmil model prediction columns.")

def metrics(df, y):
    out = []
    for c in pred_cols:
        e = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) - y
        ok = np.isfinite(e)
        e = e[ok]
        if len(e) == 0:
            continue
        out.append({
            "model": c,
            "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAE": float(np.mean(np.abs(e))),
            "R2": float(1 - np.sum(e**2) / np.sum((y[ok] - np.mean(y[ok]))**2)),
            "Pearson_r": float(np.corrcoef(y[ok], pd.to_numeric(df.loc[ok, c], errors="coerce").to_numpy())[0,1])
        })
    return pd.DataFrame(out)

# The production test predictions are the independent evaluation set.
# The 72-model architecture/descriptor comparison is therefore performed
# at the molecule level on the 129 FreeSolv test molecules.
ytest = pd.to_numeric(TEST[target], errors="coerce").to_numpy() if target in TEST.columns else None
if ytest is None:
    # test.csv from QSARmil normally has no Y_TRUE; recover it from the
    # canonical FreeSolv benchmark using the exact test SMILES.
    if "SMILES" not in TEST.columns:
        raise RuntimeError("test.csv has no Y_TRUE and no SMILES column.")
    bench = pd.read_csv(ROOT / "data" / "freesolv_benchmark.csv")
    if "smiles" not in bench.columns:
        raise RuntimeError("freesolv_benchmark.csv lacks 'smiles'.")
    lookup = dict(zip(bench["smiles"].astype(str), bench["experimental_dG_hyd_kcal_mol"]))
    ytest = TEST["SMILES"].astype(str).map(lookup).to_numpy(dtype=float)
    if np.any(~np.isfinite(ytest)):
        # Try lowercase column name if the QSARmil file uses lowercase.
        lookup = dict(zip(bench["canonical_isomeric_smiles"].astype(str),
                          bench["experimental_dG_hyd_kcal_mol"]))
        if "canonical_isomeric_smiles" in TEST.columns:
            ytest = TEST["canonical_isomeric_smiles"].astype(str).map(lookup).to_numpy(dtype=float)
    if np.any(~np.isfinite(ytest)):
        raise RuntimeError("Could not recover all 129 FreeSolv test targets.")

# Parse descriptor and architecture from exact QSARmil column names.
descriptors = [
    "RDKitGEOM", "RDKitAUTOCORR", "RDKitRDF", "RDKitMORSE", "RDKitWHIM",
    "MolFeatUSRD", "MolFeatElectroShape", "RDKitGETAWAY", "MolFeatPmapper"
]
architectures = [
    "MeanInstanceWrapperMLPNetworkRegressor",
    "MeanBagWrapperMLPNetworkRegressor",
    "MeanBagNetworkRegressor",
    "MeanInstanceNetworkRegressor",
    "AdditiveAttentionNetworkRegressor",
    "SelfAttentionNetworkRegressor",
    "HopfieldAttentionNetworkRegressor",
    "DynamicPoolingNetworkRegressor"
]

def split_model(c):
    for d in descriptors:
        prefix = d + "|"
        if c.startswith(prefix):
            a = c[len(prefix):]
            if a in architectures:
                return d, a
    return None, None

records = []
for c in pred_cols:
    d, a = split_model(c)
    if d is None:
        raise RuntimeError(f"Could not parse model column: {c}")
    records.append((d, a, c))

# Test-set metrics
rows = []
for d, a, c in records:
    pred = pd.to_numeric(TEST[c], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(pred) & np.isfinite(ytest)
    e = pred[ok] - ytest[ok]
    rows.append({
        "descriptor": d, "architecture": a, "model": c,
        "n_test": int(ok.sum()),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "R2": float(1 - np.sum(e**2) / np.sum((ytest[ok] - np.mean(ytest[ok]))**2)),
        "Pearson_r": float(np.corrcoef(ytest[ok], pred[ok])[0,1])
    })
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(OUT_DIR / "all_72_model_metrics.csv", index=False)

# Per-molecule absolute-error matrix
ae = {}
for d, a, c in records:
    pred = pd.to_numeric(TEST[c], errors="coerce").to_numpy(dtype=float)
    ae[(d, a)] = np.abs(pred - ytest)

# Descriptor paired comparisons within each architecture
desc_rows = []
for a in architectures:
    for i, d1 in enumerate(descriptors):
        for d2 in descriptors[i+1:]:
            x = ae[(d1,a)]
            y = ae[(d2,a)]
            ok = np.isfinite(x) & np.isfinite(y)
            diff = x[ok] - y[ok]
            stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto")
            desc_rows.append({
                "architecture": a,
                "descriptor_1": d1,
                "descriptor_2": d2,
                "mean_difference": float(np.mean(diff)),
                "median_difference": float(np.median(diff)),
                "fraction_d1_better": float(np.mean(diff < 0)),
                "wilcoxon_p": float(p),
                "n_pairs": int(len(diff))
            })
pd.DataFrame(desc_rows).to_csv(OUT_DIR / "paired_descriptor_comparisons.csv", index=False)

# Architecture paired comparisons within each descriptor
arch_rows = []
for d in descriptors:
    for i, a1 in enumerate(architectures):
        for a2 in architectures[i+1:]:
            x = ae[(d,a1)]
            y = ae[(d,a2)]
            ok = np.isfinite(x) & np.isfinite(y)
            diff = x[ok] - y[ok]
            stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto")
            arch_rows.append({
                "descriptor": d,
                "architecture_1": a1,
                "architecture_2": a2,
                "mean_difference": float(np.mean(diff)),
                "median_difference": float(np.median(diff)),
                "fraction_a1_better": float(np.mean(diff < 0)),
                "wilcoxon_p": float(p),
                "n_pairs": int(len(diff))
            })
pd.DataFrame(arch_rows).to_csv(OUT_DIR / "paired_architecture_comparisons.csv", index=False)

# Matrices
for metric in ["RMSE", "MAE", "R2"]:
    mat = metrics_df.pivot(index="descriptor", columns="architecture", values=metric)
    mat.to_csv(OUT_DIR / f"{metric}_matrix_5conf.csv")

# Descriptive two-way variance decomposition on the 72 aggregate RMSE cells.
m = metrics_df.copy()
grand = m["RMSE"].mean()
desc_means = m.groupby("descriptor")["RMSE"].mean()
arch_means = m.groupby("architecture")["RMSE"].mean()
ss_total = float(((m["RMSE"] - grand)**2).sum())
ss_d = sum(8 * (v-grand)**2 for v in desc_means)
ss_a = sum(9 * (v-grand)**2 for v in arch_means)
ss_i = ss_total - ss_d - ss_a
pd.DataFrame([{
    "n_conformers": 5,
    "SS_total": ss_total,
    "descriptor_fraction": ss_d/ss_total,
    "architecture_fraction": ss_a/ss_total,
    "interaction_fraction": ss_i/ss_total
}]).to_csv(OUT_DIR / "variance_decomposition_5conf.csv", index=False)

print("\n" + "="*80)
print("FREESOLV 5-CONFORMER DESCRIPTOR × ARCHITECTURE ANALYSIS")
print("="*80)
print(f"Test molecules: {len(TEST)}")
print(f"Models: {len(pred_cols)}")
print("\nBest 10 individual models by test RMSE:")
print(metrics_df.sort_values("RMSE").head(10)[["descriptor","architecture","RMSE","MAE","R2"]].to_string(index=False))
print("\nVariance decomposition:")
print(pd.read_csv(OUT_DIR / "variance_decomposition_5conf.csv").to_string(index=False))
print("\nOutputs:")
print(OUT_DIR)

