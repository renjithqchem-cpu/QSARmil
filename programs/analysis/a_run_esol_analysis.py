from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# ---------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------
BASE = Path.home() / "Documents" / "qsarmil" / "data"

ESOL_FILE = BASE / "delaney-processed.csv"
SIL_FILE = BASE / "esol_sil_predictions.csv"

QSARMIL_FILES = {
    5: BASE / "esol_qsarmil_5conf_predictions.csv",
    10: BASE / "esol_qsarmil_10conf_predictions.csv",
    15: BASE / "esol_qsarmil_15conf_predictions.csv",
    20: BASE / "esol_qsarmil_20conf_predictions.csv",
}

OUT = BASE / "esol_analysis"
FIG = OUT / "figures"

FLEX_ORDER = ["0", "1-3", "4-6", ">=7"]

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_col(df, candidates):
    nmap = {norm(c): c for c in df.columns}
    # exact normalized match first
    for c in candidates:
        if norm(c) in nmap:
            return nmap[norm(c)]
    # then substring match
    for c in df.columns:
        nc = norm(c)
        for cand in candidates:
            x = norm(cand)
            if x in nc or nc in x:
                return c
    return None

def detect_target(df):
    candidates = [
        "measured log solubility in mols per litre",
        "measured log solubility",
        "y_true", "true", "target", "actual",
        "measured", "experimental", "solubility",
    ]
    c = find_col(df, candidates)
    if c is not None:
        return c
    # Numeric column with common target-like names
    for c in df.columns:
        nc = norm(c)
        if "solubility" in nc and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def detect_prediction(df):
    candidates = [
        "prediction", "predicted", "y_pred", "ypred",
        "test_prediction", "consensus_prediction", "pred",
    ]
    c = find_col(df, candidates)
    if c is not None:
        return c

    # Avoid the original measured/ESOL-predicted columns if possible.
    bad = {"smiles", "compoundid", "compound", "target", "actual"}
    numeric = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and norm(c) not in bad:
            nc = norm(c)
            if "measured" not in nc and "true" not in nc and "target" not in nc:
                numeric.append(c)
    # If there is exactly one plausible numeric column, use it.
    if len(numeric) == 1:
        return numeric[0]
    return None

def detect_smiles(df):
    return find_col(df, ["smiles", "canonical_smiles", "mol_smiles"])

def detect_id(df):
    return find_col(df, ["compound id", "compound_id", "id", "mol_id", "name"])

def flex_group(rotb):
    if pd.isna(rotb):
        return "unknown"
    if rotb == 0:
        return "0"
    if rotb <= 3:
        return "1-3"
    if rotb <= 6:
        return "4-6"
    return ">=7"

def metrics(y, p):
    mask = np.isfinite(y) & np.isfinite(p)
    y = np.asarray(y)[mask]
    p = np.asarray(p)[mask]
    if len(y) == 0:
        return dict(N=0, R2=np.nan, RMSE=np.nan, MAE=np.nan, Pearson_r=np.nan)
    r = pearsonr(y, p).statistic if len(y) >= 2 else np.nan
    return dict(
        N=len(y),
        R2=r2_score(y, p),
        RMSE=np.sqrt(mean_squared_error(y, p)),
        MAE=mean_absolute_error(y, p),
        Pearson_r=r,
    )

def load_prediction(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    print(f"\n[{label}] {path}")
    print(f"  shape = {df.shape}")
    print(f"  columns = {list(df.columns)}")

    target = detect_target(df)
    pred = detect_prediction(df)
    smiles = detect_smiles(df)
    cid = detect_id(df)

    print(f"  detected target = {target}")
    print(f"  detected prediction = {pred}")
    print(f"  detected smiles = {smiles}")
    print(f"  detected id = {cid}")

    if target is None or pred is None:
        raise RuntimeError(
            f"Could not detect target/prediction columns in {path}. "
            f"Please inspect the printed columns and edit detect_target()/"
            f"detect_prediction() if necessary."
        )

    out = pd.DataFrame({
        "target": pd.to_numeric(df[target], errors="coerce"),
        "prediction": pd.to_numeric(df[pred], errors="coerce"),
    })

    if smiles:
        out["smiles"] = df[smiles].astype(str)
    if cid:
        out["compound_id"] = df[cid].astype(str)

    # Preserve original row order; this matters when prediction files
    # correspond to the fixed ESOL test split.
    out["_row"] = np.arange(len(out))
    return out, df, target, pred

def align_with_esol(pred_df, esol):
    """Prefer SMILES/ID alignment; otherwise require equal row count."""
    esol_smiles = detect_smiles(esol)
    esol_id = detect_id(esol)

    if "smiles" in pred_df.columns and esol_smiles:
        # duplicate SMILES are possible; use merge with validation only
        # if unique, otherwise fall back to row alignment.
        if pred_df["smiles"].is_unique and esol[esol_smiles].astype(str).is_unique:
            tmp = esol.copy()
            tmp["_esol_smiles"] = tmp[esol_smiles].astype(str)
            m = pred_df.merge(
                tmp[["_esol_smiles", "Number of Rotatable Bonds"]],
                left_on="smiles", right_on="_esol_smiles", how="left"
            )
            if m["Number of Rotatable Bonds"].notna().sum() >= 0.95 * len(m):
                return m["Number of Rotatable Bonds"].to_numpy()

    if "compound_id" in pred_df.columns and esol_id:
        if pred_df["compound_id"].is_unique and esol[esol_id].astype(str).is_unique:
            tmp = esol.copy()
            tmp["_esol_id"] = tmp[esol_id].astype(str)
            m = pred_df.merge(
                tmp[["_esol_id", "Number of Rotatable Bonds"]],
                left_on="compound_id", right_on="_esol_id", how="left"
            )
            if m["Number of Rotatable Bonds"].notna().sum() >= 0.95 * len(m):
                return m["Number of Rotatable Bonds"].to_numpy()

    # Final fallback: row order. This is valid for the known fixed
    # QSARmil/SIL ESOL test files if they were produced from the same split.
    if len(pred_df) == 226:
        test = esol.iloc[0:0].copy()
        # Recreate the exact split used in the project:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(esol))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.20, random_state=42
        )
        rotb = esol.iloc[test_idx]["Number of Rotatable Bonds"].to_numpy()
        return rotb

    raise RuntimeError(
        "Could not align predictions to ESOL rotatable-bond values. "
        "Use prediction files containing SMILES or Compound ID."
    )

def safe_model_columns(raw_df, pred_col, target_col):
    """Return numeric columns that look like individual model predictions."""
    cols = []
    for c in raw_df.columns:
        if c in {pred_col, target_col}:
            continue
        if not pd.api.types.is_numeric_dtype(raw_df[c]):
            continue
        nc = norm(c)
        if any(x in nc for x in ["prediction", "predicted", "ypred"]):
            cols.append(c)
        elif "|" in str(c):
            cols.append(c)
    return cols

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

esol = pd.read_csv(ESOL_FILE)
print("=" * 80)
print("ESOL CONFORMER-SCALING ANALYSIS")
print("=" * 80)
print(f"ESOL dataset: {ESOL_FILE}")
print(f"shape: {esol.shape}")

all_results = {}
raw_results = {}
detected = {}

sil, sil_raw, sil_target, sil_pred = load_prediction(SIL_FILE, "SIL")
all_results["SIL"] = sil
raw_results["SIL"] = sil_raw
detected["SIL"] = (sil_target, sil_pred)

for nconf, path in QSARMIL_FILES.items():
    d, raw, tc, pc = load_prediction(path, f"QSARmil {nconf} conf")
    all_results[str(nconf)] = d
    raw_results[str(nconf)] = raw
    detected[str(nconf)] = (tc, pc)

# ---------------------------------------------------------------------
# Add flexibility groups
# ---------------------------------------------------------------------
for label, df in all_results.items():
    rotb = align_with_esol(df, esol)
    df["rotatable_bonds"] = rotb
    df["flexibility"] = pd.Series(rotb).map(flex_group).to_numpy()

# ---------------------------------------------------------------------
# Overall metrics
# ---------------------------------------------------------------------
rows = []
for label, df in all_results.items():
    m = metrics(df["target"], df["prediction"])
    m["Model"] = label
    m["Conformers"] = 1 if label == "SIL" else int(label)
    rows.append(m)

overall = pd.DataFrame(rows)[
    ["Model", "Conformers", "N", "R2", "RMSE", "MAE", "Pearson_r"]
]
overall.to_csv(OUT / "overall_metrics.csv", index=False)

# ---------------------------------------------------------------------
# Improvement vs SIL
# Positive Delta_R2 / Delta_r and positive Delta_RMSE / Delta_MAE mean
# improvement in the respective direction.
# ---------------------------------------------------------------------
sil_m = overall[overall["Model"] == "SIL"].iloc[0]
imp = overall[overall["Model"] != "SIL"].copy()
imp["Delta_R2"] = imp["R2"] - sil_m["R2"]
imp["Delta_RMSE"] = sil_m["RMSE"] - imp["RMSE"]
imp["Delta_MAE"] = sil_m["MAE"] - imp["MAE"]
imp["Delta_Pearson_r"] = imp["Pearson_r"] - sil_m["Pearson_r"]
imp = imp[
    ["Model", "Conformers", "Delta_R2", "Delta_RMSE",
     "Delta_MAE", "Delta_Pearson_r"]
]
imp.to_csv(OUT / "improvement_vs_sil.csv", index=False)

# ---------------------------------------------------------------------
# Flexibility-stratified metrics
# ---------------------------------------------------------------------
flex_rows = []
for label, df in all_results.items():
    for fg in FLEX_ORDER:
        sub = df[df["flexibility"] == fg]
        if len(sub) == 0:
            continue
        m = metrics(sub["target"], sub["prediction"])
        flex_rows.append({
            "Model": label,
            "Conformers": 1 if label == "SIL" else int(label),
            "Flexibility": fg,
            **m
        })

flex = pd.DataFrame(flex_rows)
flex.to_csv(OUT / "flexibility_metrics.csv", index=False)

# ---------------------------------------------------------------------
# Prediction changes relative to SIL and consecutive conformer counts
# ---------------------------------------------------------------------
base = all_results["SIL"].copy()
change_rows = []

for nconf in [5, 10, 15, 20]:
    d = all_results[str(nconf)].copy()
    if len(d) != len(base):
        raise RuntimeError(f"SIL and {nconf}-conf prediction lengths differ.")
    change = d["prediction"].to_numpy() - base["prediction"].to_numpy()
    abs_err_sil = np.abs(base["target"].to_numpy() - base["prediction"].to_numpy())
    abs_err_mil = np.abs(d["target"].to_numpy() - d["prediction"].to_numpy())

    for i in range(len(d)):
        change_rows.append({
            "row": i,
            "conformers": nconf,
            "target": d.iloc[i]["target"],
            "rotatable_bonds": d.iloc[i]["rotatable_bonds"],
            "flexibility": d.iloc[i]["flexibility"],
            "SIL_prediction": base.iloc[i]["prediction"],
            "MIL_prediction": d.iloc[i]["prediction"],
            "prediction_change_MIL_minus_SIL": change[i],
            "abs_prediction_change": abs(change[i]),
            "abs_error_SIL": abs_err_sil[i],
            "abs_error_MIL": abs_err_mil[i],
            "error_improvement": abs_err_sil[i] - abs_err_mil[i],
        })

changes = pd.DataFrame(change_rows)
changes.to_csv(OUT / "prediction_changes.csv", index=False)

# ---------------------------------------------------------------------
# Scaling table
# ---------------------------------------------------------------------
scaling = overall.sort_values("Conformers").copy()
scaling["Delta_R2_vs_SIL"] = scaling["R2"] - sil_m["R2"]
scaling["Delta_RMSE_vs_SIL"] = sil_m["RMSE"] - scaling["RMSE"]
scaling["Delta_MAE_vs_SIL"] = sil_m["MAE"] - scaling["MAE"]
scaling["Delta_r_vs_SIL"] = scaling["Pearson_r"] - sil_m["Pearson_r"]
scaling.to_csv(OUT / "conformer_scaling.csv", index=False)

# ---------------------------------------------------------------------
# Consensus/model-column inspection
# ---------------------------------------------------------------------
cons_rows = []
for label, raw in raw_results.items():
    tc, pc = detected[label]
    model_cols = safe_model_columns(raw, pc, tc)
    for c in model_cols:
        vals = pd.to_numeric(raw[c], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        cons_rows.append({
            "Dataset": label,
            "Model_column": c,
            "N_valid": vals.notna().sum(),
            "Mean_prediction": vals.mean(),
            "Std_prediction": vals.std(),
        })

cons = pd.DataFrame(cons_rows)
if len(cons):
    cons.to_csv(OUT / "consensus_models.csv", index=False)
else:
    pd.DataFrame(columns=[
        "Dataset", "Model_column", "N_valid",
        "Mean_prediction", "Std_prediction"
    ]).to_csv(OUT / "consensus_models.csv", index=False)

# ---------------------------------------------------------------------
# FIGURE 1: performance vs conformer count
# ---------------------------------------------------------------------
qs = scaling[scaling["Model"] != "SIL"].sort_values("Conformers")

plt.figure(figsize=(7, 5))
plt.plot(qs["Conformers"], qs["R2"], marker="o", label="R²")
plt.plot(qs["Conformers"], qs["Pearson_r"], marker="o", label="Pearson r")
plt.axhline(sil_m["R2"], linestyle="--", label="SIL R²")
plt.axhline(sil_m["Pearson_r"], linestyle=":", label="SIL Pearson r")
plt.xlabel("Number of conformers")
plt.ylabel("Metric")
plt.title("ESOL: predictive performance vs conformer count")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "performance_vs_conformers.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# FIGURE 2: error improvement vs SIL
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(qs["Conformers"], qs["Delta_RMSE_vs_SIL"], marker="o", label="RMSE improvement")
plt.plot(qs["Conformers"], qs["Delta_MAE_vs_SIL"], marker="o", label="MAE improvement")
plt.axhline(0, linestyle="--")
plt.xlabel("Number of conformers")
plt.ylabel("Improvement vs SIL")
plt.title("ESOL: error improvement relative to conventional SIL")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "delta_rmse_vs_conformers.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# FIGURE 3: RMSE by flexibility
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
for label in ["SIL", "5", "10", "15", "20"]:
    sub = flex[flex["Model"] == label].copy()
    if len(sub):
        sub["Flexibility"] = pd.Categorical(sub["Flexibility"], FLEX_ORDER, ordered=True)
        sub = sub.sort_values("Flexibility")
        plt.plot(sub["Flexibility"].astype(str), sub["RMSE"], marker="o", label=label)
plt.xlabel("Rotatable-bond group")
plt.ylabel("RMSE")
plt.title("ESOL: prediction error across molecular flexibility")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(FIG / "performance_by_flexibility.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# FIGURE 4: absolute prediction change vs flexibility
# ---------------------------------------------------------------------
grp = changes.groupby(["conformers", "flexibility"], observed=True)["abs_prediction_change"].mean().reset_index()

plt.figure(figsize=(8, 5))
for nconf in [5, 10, 15, 20]:
    sub = grp[grp["conformers"] == nconf]
    if len(sub):
        sub = sub.set_index("flexibility").reindex(FLEX_ORDER).reset_index()
        plt.plot(sub["flexibility"], sub["abs_prediction_change"], marker="o", label=f"{nconf} conf")
plt.xlabel("Rotatable-bond group")
plt.ylabel("Mean |MIL − SIL prediction|")
plt.title("ESOL: conformer-based prediction changes vs flexibility")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "prediction_change_vs_flexibility.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# TEXT REPORT
# ---------------------------------------------------------------------
lines = []
lines.append("=" * 80)
lines.append("ESOL CONFORMER-SCALING ANALYSIS REPORT")
lines.append("=" * 80)
lines.append("")
lines.append("Dataset: Delaney ESOL")
lines.append(f"Dataset rows: {len(esol)}")
lines.append("Fixed project split: random_state=42, test_size=0.20")
lines.append("")
lines.append("IMPORTANT: This is a descriptive analysis of the completed runs.")
lines.append("It does not establish statistical significance or an optimal conformer")
lines.append("count from a single train/test split.")
lines.append("")

lines.append("OVERALL PERFORMANCE")
lines.append("-" * 80)
lines.append(overall.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
lines.append("")

lines.append("IMPROVEMENT RELATIVE TO SIL")
lines.append("-" * 80)
lines.append("Delta_R2 = R2(MIL) - R2(SIL)")
lines.append("Delta_RMSE = RMSE(SIL) - RMSE(MIL)")
lines.append("Delta_MAE = MAE(SIL) - MAE(MIL)")
lines.append("Delta_Pearson_r = r(MIL) - r(SIL)")
lines.append(imp.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
lines.append("")

lines.append("FLEXIBILITY-STRATIFIED PERFORMANCE")
lines.append("-" * 80)
lines.append(flex.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
lines.append("")

lines.append("CURRENT DESCRIPTIVE OBSERVATION")
lines.append("-" * 80)
best_r2 = qs.loc[qs["R2"].idxmax()]
best_rmse = qs.loc[qs["RMSE"].idxmin()]
best_mae = qs.loc[qs["MAE"].idxmin()]
lines.append(
    f"Among 5/10/15/20 conformers, the highest R2 is at "
    f"{int(best_r2['Conformers'])} conformers (R2={best_r2['R2']:.6f})."
)
lines.append(
    f"Among 5/10/15/20 conformers, the lowest RMSE is at "
    f"{int(best_rmse['Conformers'])} conformers (RMSE={best_rmse['RMSE']:.6f})."
)
lines.append(
    f"Among 5/10/15/20 conformers, the lowest MAE is at "
    f"{int(best_mae['Conformers'])} conformers (MAE={best_mae['MAE']:.6f})."
)
lines.append("")
lines.append(
    "The 5->10->15->20 trajectory should be described as non-monotonic "
    "unless additional replicated splits/seeds demonstrate otherwise."
)
lines.append(
    "A 25-conformer run should therefore be treated as another point in "
    "the scaling experiment, not as a presumed optimum."
)
lines.append("")

lines.append("FILES CREATED")
lines.append("-" * 80)
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        lines.append(str(p))

(OUT / "ESOL_ANALYSIS_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"Output directory: {OUT}")
print(f"Overall metrics: {OUT/'overall_metrics.csv'}")
print(f"Flexibility metrics: {OUT/'flexibility_metrics.csv'}")
print(f"Figures: {FIG}")
print(f"Report: {OUT/'ESOL_ANALYSIS_REPORT.txt'}")
print("\nKey descriptive results:")
print(overall.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\nDo NOT interpret the best row as an established optimum; this is one split.")
