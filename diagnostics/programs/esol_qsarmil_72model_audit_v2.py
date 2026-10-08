from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

BASE = Path.home() / "Documents" / "qsarmil"
DATA = BASE / "data"
OUT = DATA / "esol_72model_audit"
OUT.mkdir(parents=True, exist_ok=True)

CONFS = [5, 10, 15, 20]
ESOL_FILE = DATA / "delaney-processed.csv"
TARGET = "measured log solubility in mols per litre"

# ---------------------------------------------------------------------
# Load original ESOL data
# ---------------------------------------------------------------------
print("=" * 90)
print("ESOL QSARMIL 72-MODEL AUDIT")
print("=" * 90)
print("Target is recovered from the original Delaney ESOL dataset.")
print(f"ESOL file: {ESOL_FILE}")

if not ESOL_FILE.exists():
    raise FileNotFoundError(f"Missing original ESOL file: {ESOL_FILE}")

esol = pd.read_csv(ESOL_FILE)

if "smiles" not in esol.columns:
    raise ValueError("Original ESOL file does not contain 'smiles'.")

if TARGET not in esol.columns:
    raise ValueError(f"Original ESOL file does not contain target: {TARGET}")

# Normalize only for matching; retain original values.
esol["_SMILES_KEY"] = esol["smiles"].astype(str).str.strip()

# Check duplicate SMILES in source data.
dup = esol["_SMILES_KEY"].duplicated(keep=False).sum()
if dup:
    print(f"WARNING: {dup} rows belong to duplicated SMILES in original ESOL data.")
    print("The first occurrence will be used for target matching.")

target_lookup = (
    esol.drop_duplicates("_SMILES_KEY", keep="first")
        .set_index("_SMILES_KEY")[TARGET]
)

records = []
all_prediction_tables = {}

# ---------------------------------------------------------------------
# Process each conformer count
# ---------------------------------------------------------------------
for conf in CONFS:
    f = DATA / f"esol_qsarmil_{conf}conf" / "test.csv"

    print("\n" + "-" * 90)
    print(f"Reading: {f}")

    if not f.exists():
        print("  MISSING -- skipped")
        continue

    d = pd.read_csv(f)
    d.columns = [str(c).strip() for c in d.columns]

    if "SMILES" not in d.columns:
        raise ValueError(f"{f} does not contain SMILES column.")

    model_cols = [c for c in d.columns if "|" in str(c)]

    print(f"  rows = {len(d)}")
    print(f"  model columns = {len(model_cols)}")

    if len(model_cols) != 72:
        raise ValueError(
            f"{f}: expected 72 model columns, found {len(model_cols)}"
        )

    keys = d["SMILES"].astype(str).str.strip()
    y = keys.map(target_lookup)

    unmatched = y.isna().sum()
    if unmatched:
        print(f"  WARNING: {unmatched} SMILES did not match original ESOL data.")
        print("  Attempting canonical RDKit matching...")

        try:
            from rdkit import Chem

            def canonical(s):
                m = Chem.MolFromSmiles(s)
                return Chem.MolToSmiles(m, canonical=True) if m else None

            source_can = {}
            for smi, val in zip(esol["smiles"].astype(str), esol[TARGET]):
                cs = canonical(smi)
                if cs is not None and cs not in source_can:
                    source_can[cs] = val

            y2 = keys.map(lambda s: source_can.get(canonical(s)))
            y = y.fillna(y2)
            unmatched = y.isna().sum()
            print(f"  Remaining unmatched after canonical matching = {unmatched}")

        except Exception as exc:
            print(f"  RDKit fallback failed: {exc}")

    if unmatched:
        raise ValueError(
            f"Could not recover target values for {unmatched} rows in {f}."
        )

    y = pd.to_numeric(y, errors="coerce").to_numpy()

    # Save the matched target + predictions for traceability.
    matched = pd.DataFrame({
        "SMILES": d["SMILES"],
        "Y_TRUE_ESOL": y
    })
    for c in model_cols:
        matched[c] = d[c]
    matched.to_csv(OUT / f"matched_test_predictions_{conf}conf.csv", index=False)

    all_prediction_tables[conf] = matched

    for model in model_cols:
        pred = pd.to_numeric(d[model], errors="coerce").to_numpy()
        ok = np.isfinite(y) & np.isfinite(pred)

        yy = y[ok]
        pp = pred[ok]

        if len(yy) < 2:
            continue

        r, pval = pearsonr(yy, pp)

        descriptor, algorithm = str(model).split("|", 1)

        records.append({
            "Conformers": conf,
            "Descriptor": descriptor,
            "Model": algorithm,
            "Model_full": model,
            "N": len(yy),
            "R2": r2_score(yy, pp),
            "RMSE": np.sqrt(mean_squared_error(yy, pp)),
            "MAE": mean_absolute_error(yy, pp),
            "Pearson_r": r,
            "Pearson_p": pval
        })

audit = pd.DataFrame(records)

if audit.empty:
    raise RuntimeError("No model-level records recovered.")

# ---------------------------------------------------------------------
# Main model table
# ---------------------------------------------------------------------
audit = audit.sort_values(
    ["Conformers", "RMSE", "MAE", "R2"],
    ascending=[True, True, True, False]
)

audit.to_csv(OUT / "all_72_models.csv", index=False)

# ---------------------------------------------------------------------
# Rankings within each conformer count
# ---------------------------------------------------------------------
ranked = audit.copy()
ranked["RMSE_rank"] = ranked.groupby("Conformers")["RMSE"].rank(
    method="min", ascending=True
)
ranked["MAE_rank"] = ranked.groupby("Conformers")["MAE"].rank(
    method="min", ascending=True
)
ranked["R2_rank"] = ranked.groupby("Conformers")["R2"].rank(
    method="min", ascending=False
)
ranked["Mean_rank"] = ranked[
    ["RMSE_rank", "MAE_rank", "R2_rank"]
].mean(axis=1)

ranked.to_csv(OUT / "all_72_models_ranked.csv", index=False)

# ---------------------------------------------------------------------
# Top models per conformer count -- descriptive only
# ---------------------------------------------------------------------
best_rows = []

for conf in CONFS:
    s = audit[audit["Conformers"] == conf]
    if s.empty:
        continue

    for criterion, idx in [
        ("RMSE", s["RMSE"].idxmin()),
        ("MAE", s["MAE"].idxmin()),
        ("R2", s["R2"].idxmax())
    ]:
        x = s.loc[idx]
        best_rows.append({
            "Conformers": conf,
            "Criterion": criterion,
            "Descriptor": x["Descriptor"],
            "Model": x["Model"],
            "Model_full": x["Model_full"],
            "R2": x["R2"],
            "RMSE": x["RMSE"],
            "MAE": x["MAE"],
            "Pearson_r": x["Pearson_r"]
        })

best = pd.DataFrame(best_rows)
best.to_csv(OUT / "best_models_by_conformer.csv", index=False)

# ---------------------------------------------------------------------
# Top 10 by RMSE at each conformer count
# ---------------------------------------------------------------------
top10_rows = []

for conf in CONFS:
    s = audit[audit["Conformers"] == conf].sort_values("RMSE").head(10)

    for rank, (_, x) in enumerate(s.iterrows(), 1):
        top10_rows.append({
            "Conformers": conf,
            "RMSE_rank": rank,
            "Descriptor": x["Descriptor"],
            "Model": x["Model"],
            "Model_full": x["Model_full"],
            "R2": x["R2"],
            "RMSE": x["RMSE"],
            "MAE": x["MAE"],
            "Pearson_r": x["Pearson_r"]
        })

top10 = pd.DataFrame(top10_rows)
top10.to_csv(OUT / "top10_RMSE_each_conformer.csv", index=False)

counts = (
    top10.groupby(["Descriptor", "Model", "Model_full"])
    .agg(
        top10_appearances=("Conformers", "count"),
        conformer_settings=(
            "Conformers",
            lambda x: ",".join(map(str, sorted(x)))
        )
    )
    .reset_index()
    .sort_values(
        ["top10_appearances", "Descriptor", "Model"],
        ascending=[False, True, True]
    )
)

counts.to_csv(OUT / "top10_model_appearance_counts.csv", index=False)

# ---------------------------------------------------------------------
# Exact model stability across 5/10/15/20 conformers
# ---------------------------------------------------------------------
stability = audit.pivot_table(
    index=["Descriptor", "Model", "Model_full"],
    columns="Conformers",
    values=["R2", "RMSE", "MAE", "Pearson_r"],
    aggfunc="first"
)

stability.columns = [
    f"{metric}_{conf}conf"
    for metric, conf in stability.columns
]
stability = stability.reset_index()

expected = [
    f"{metric}_{conf}conf"
    for metric in ["R2", "RMSE", "MAE", "Pearson_r"]
    for conf in CONFS
]

complete = stability.dropna(subset=expected).copy()

if not complete.empty:
    complete["mean_RMSE"] = complete[
        [f"RMSE_{c}conf" for c in CONFS]
    ].mean(axis=1)

    complete["sd_RMSE"] = complete[
        [f"RMSE_{c}conf" for c in CONFS]
    ].std(axis=1)

    complete["range_RMSE"] = (
        complete[[f"RMSE_{c}conf" for c in CONFS]].max(axis=1)
        - complete[[f"RMSE_{c}conf" for c in CONFS]].min(axis=1)
    )

    complete["mean_MAE"] = complete[
        [f"MAE_{c}conf" for c in CONFS]
    ].mean(axis=1)

    complete["sd_MAE"] = complete[
        [f"MAE_{c}conf" for c in CONFS]
    ].std(axis=1)

    complete["mean_R2"] = complete[
        [f"R2_{c}conf" for c in CONFS]
    ].mean(axis=1)

    complete["sd_R2"] = complete[
        [f"R2_{c}conf" for c in CONFS]
    ].std(axis=1)

    complete = complete.sort_values(
        ["mean_RMSE", "mean_MAE", "sd_RMSE"]
    )

complete.to_csv(OUT / "model_stability_across_conformers.csv", index=False)

# ---------------------------------------------------------------------
# Cross-conformer summary
# ---------------------------------------------------------------------
summary = (
    audit.groupby(["Descriptor", "Model", "Model_full"])
    .agg(
        conformer_settings=("Conformers", "nunique"),
        mean_RMSE=("RMSE", "mean"),
        median_RMSE=("RMSE", "median"),
        sd_RMSE=("RMSE", "std"),
        mean_MAE=("MAE", "mean"),
        median_MAE=("MAE", "median"),
        mean_R2=("R2", "mean"),
        median_R2=("R2", "median"),
        mean_Pearson_r=("Pearson_r", "mean")
    )
    .reset_index()
    .sort_values(["mean_RMSE", "mean_MAE"])
)

summary.to_csv(OUT / "model_cross_conformer_summary.csv", index=False)

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
lines = []
lines.append("=" * 90)
lines.append("ESOL QSARMIL 72-MODEL AUDIT")
lines.append("=" * 90)
lines.append("Existing QSARmil prediction files were analyzed.")
lines.append("No new QSARmil calculations were performed.")
lines.append("")
lines.append(f"Original ESOL target file: {ESOL_FILE}")
lines.append(f"Target: {TARGET}")
lines.append("Target matching: QSARmil test.csv SMILES -> original ESOL SMILES")
lines.append("")

lines.append("MODEL RECORDS")
lines.append(
    audit.groupby("Conformers").size()
    .rename("N_model_records")
    .to_string()
)
lines.append("")

lines.append("BEST INDIVIDUAL MODEL PER CONFORMER SETTING")
lines.append(best.to_string(index=False))
lines.append("")

lines.append("TOP 10 MODELS BY RMSE")
lines.append(top10.to_string(index=False))
lines.append("")

lines.append("TOP-10 APPEARANCE COUNTS")
lines.append(counts.to_string(index=False))
lines.append("")

lines.append("CROSS-CONFORMER MODEL SUMMARY")
lines.append(summary.to_string(index=False))
lines.append("")

lines.append("SCIENTIFIC CAUTION")
lines.append(
    "These rankings are descriptive. Because all models are evaluated on the same "
    "test set, the test results should not be repeatedly used to select a final "
    "publication model without an independent validation/replication protocol."
)
lines.append(
    "The analysis is intended to determine model behavior and stability, not to "
    "claim that a single architecture is universally superior."
)

(OUT / "ESOL_72_MODEL_AUDIT_REPORT.txt").write_text(
    "\n".join(lines), encoding="utf-8"
)

print("\n" + "=" * 90)
print("AUDIT COMPLETE")
print("=" * 90)
print(f"Output directory: {OUT}")
print(f"Total model records: {len(audit)}")
print("\nRecords by conformer count:")
print(audit.groupby("Conformers").size())

print("\nTop 3 by RMSE for each conformer count:")
for conf in CONFS:
    s = audit[audit["Conformers"] == conf].sort_values("RMSE").head(3)
    print(f"\n{conf} conformers:")
    print(s[["Descriptor", "Model", "R2", "RMSE", "MAE"]].to_string(index=False))

print("\nFiles written:")
for f in sorted(OUT.iterdir()):
    print(" ", f.name)

print("\nIMPORTANT: no new QSARmil calculations were performed.")
