from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path.home() / "Documents" / "qsarmil"
AUDIT = BASE / "data" / "esol_72model_audit"
OUT = BASE / "data" / "esol_288_model_analysis"
OUT.mkdir(parents=True, exist_ok=True)

INPUT = AUDIT / "all_72_models.csv"
CONFS = [5, 10, 15, 20]

if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing {INPUT}\nRun esol_qsarmil_72model_audit_v2.py first."
    )

df = pd.read_csv(INPUT)

required = [
    "Conformers", "Descriptor", "Model", "Model_full",
    "N", "R2", "RMSE", "MAE", "Pearson_r"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

print("=" * 100)
print("ESOL QSARMIL — 288-MODEL STATISTICAL ANALYSIS")
print("=" * 100)
print(f"Input: {INPUT}")
print(f"Records: {len(df)}")

# ------------------------------------------------------------
# 1. Descriptor-level summary
# ------------------------------------------------------------
descriptor = (
    df.groupby("Descriptor")
      .agg(
          n_evaluations=("Model_full", "count"),
          mean_RMSE=("RMSE", "mean"),
          median_RMSE=("RMSE", "median"),
          sd_RMSE=("RMSE", "std"),
          min_RMSE=("RMSE", "min"),
          mean_MAE=("MAE", "mean"),
          median_MAE=("MAE", "median"),
          mean_R2=("R2", "mean"),
          median_R2=("R2", "median"),
          mean_Pearson_r=("Pearson_r", "mean")
      )
      .reset_index()
      .sort_values("mean_RMSE")
)
descriptor.to_csv(OUT / "descriptor_summary.csv", index=False)

# ------------------------------------------------------------
# 2. Architecture-level summary
# ------------------------------------------------------------
architecture = (
    df.groupby("Model")
      .agg(
          n_evaluations=("Model_full", "count"),
          mean_RMSE=("RMSE", "mean"),
          median_RMSE=("RMSE", "median"),
          sd_RMSE=("RMSE", "std"),
          min_RMSE=("RMSE", "min"),
          mean_MAE=("MAE", "mean"),
          median_MAE=("MAE", "median"),
          mean_R2=("R2", "mean"),
          median_R2=("R2", "median"),
          mean_Pearson_r=("Pearson_r", "mean")
      )
      .reset_index()
      .sort_values("mean_RMSE")
)
architecture.to_csv(OUT / "architecture_summary.csv", index=False)

# ------------------------------------------------------------
# 3. Conformer-level summary across all 72 models
# ------------------------------------------------------------
conformer = (
    df.groupby("Conformers")
      .agg(
          n_models=("Model_full", "count"),
          mean_RMSE=("RMSE", "mean"),
          median_RMSE=("RMSE", "median"),
          sd_RMSE=("RMSE", "std"),
          min_RMSE=("RMSE", "min"),
          mean_MAE=("MAE", "mean"),
          median_MAE=("MAE", "median"),
          mean_R2=("R2", "mean"),
          median_R2=("R2", "median"),
          mean_Pearson_r=("Pearson_r", "mean")
      )
      .reset_index()
      .sort_values("Conformers")
)
conformer.to_csv(OUT / "conformer_summary.csv", index=False)

# ------------------------------------------------------------
# 4. Descriptor × architecture summary
# ------------------------------------------------------------
da = (
    df.groupby(["Descriptor", "Model"])
      .agg(
          n_conformers=("Conformers", "nunique"),
          mean_RMSE=("RMSE", "mean"),
          median_RMSE=("RMSE", "median"),
          sd_RMSE=("RMSE", "std"),
          min_RMSE=("RMSE", "min"),
          mean_MAE=("MAE", "mean"),
          median_MAE=("MAE", "median"),
          mean_R2=("R2", "mean"),
          median_R2=("R2", "median"),
          mean_Pearson_r=("Pearson_r", "mean")
      )
      .reset_index()
      .sort_values(["mean_RMSE", "mean_MAE"])
)
da.to_csv(OUT / "descriptor_architecture_summary.csv", index=False)

# ------------------------------------------------------------
# 5. Rank all 288 individual models within conformer setting
# ------------------------------------------------------------
ranked = df.copy()
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

ranked = ranked.sort_values(
    ["Conformers", "RMSE", "MAE", "R2"],
    ascending=[True, True, True, False]
)
ranked.to_csv(OUT / "model_288_ranked.csv", index=False)

# ------------------------------------------------------------
# 6. Top-10 recurrence
# ------------------------------------------------------------
top10_parts = []
for conf in CONFS:
    s = ranked[ranked["Conformers"] == conf].sort_values("RMSE").head(10).copy()
    s["Top10_RMSE_rank"] = np.arange(1, len(s) + 1)
    top10_parts.append(s)

top10 = pd.concat(top10_parts, ignore_index=True)
top10.to_csv(OUT / "top10_recurrence.csv", index=False)

recurrence = (
    top10.groupby(["Descriptor", "Model", "Model_full"])
         .agg(
             top10_appearances=("Conformers", "count"),
             conformers_in_top10=(
                 "Conformers",
                 lambda x: ",".join(map(str, sorted(x)))
             ),
             best_RMSE=("RMSE", "min"),
             mean_top10_RMSE=("RMSE", "mean")
         )
         .reset_index()
         .sort_values(
             ["top10_appearances", "mean_top10_RMSE"],
             ascending=[False, True]
         )
)
recurrence.to_csv(OUT / "top10_recurrence_summary.csv", index=False)

# ------------------------------------------------------------
# 7. Exact model trajectories across conformer counts
# ------------------------------------------------------------
traj = df.pivot_table(
    index=["Descriptor", "Model", "Model_full"],
    columns="Conformers",
    values=["RMSE", "MAE", "R2", "Pearson_r"],
    aggfunc="first"
)
traj.columns = [f"{metric}_{conf}conf" for metric, conf in traj.columns]
traj = traj.reset_index()

for c in CONFS:
    for metric in ["RMSE", "MAE", "R2", "Pearson_r"]:
        col = f"{metric}_{c}conf"
        if col not in traj.columns:
            traj[col] = np.nan

# Explicit changes: positive ΔRMSE means worse; negative means improvement.
traj["Delta_RMSE_5_to_10"] = traj["RMSE_10conf"] - traj["RMSE_5conf"]
traj["Delta_RMSE_10_to_15"] = traj["RMSE_15conf"] - traj["RMSE_10conf"]
traj["Delta_RMSE_15_to_20"] = traj["RMSE_20conf"] - traj["RMSE_15conf"]
traj["Delta_RMSE_5_to_20"] = traj["RMSE_20conf"] - traj["RMSE_5conf"]

traj["Delta_MAE_5_to_10"] = traj["MAE_10conf"] - traj["MAE_5conf"]
traj["Delta_MAE_10_to_15"] = traj["MAE_15conf"] - traj["MAE_10conf"]
traj["Delta_MAE_15_to_20"] = traj["MAE_20conf"] - traj["MAE_15conf"]
traj["Delta_MAE_5_to_20"] = traj["MAE_20conf"] - traj["MAE_5conf"]

traj["Delta_R2_5_to_10"] = traj["R2_10conf"] - traj["R2_5conf"]
traj["Delta_R2_10_to_15"] = traj["R2_15conf"] - traj["R2_10conf"]
traj["Delta_R2_15_to_20"] = traj["R2_20conf"] - traj["R2_15conf"]
traj["Delta_R2_5_to_20"] = traj["R2_20conf"] - traj["R2_5conf"]

traj.to_csv(OUT / "model_conformer_trajectory.csv", index=False)
traj.to_csv(OUT / "conformer_effect_by_model.csv", index=False)

# ------------------------------------------------------------
# 8. Classify exact-model 5 -> 20 behavior
# ------------------------------------------------------------
def classify(delta):
    if pd.isna(delta):
        return "incomplete"
    if delta < -0.01:
        return "improved_RMSE"
    if delta > 0.01:
        return "worsened_RMSE"
    return "approximately_stable"

traj["5_to_20_RMSE_behavior"] = traj["Delta_RMSE_5_to_20"].apply(classify)

behavior = (
    traj["5_to_20_RMSE_behavior"]
    .value_counts()
    .rename_axis("behavior")
    .reset_index(name="n_models")
)
behavior.to_csv(OUT / "conformer_behavior_counts.csv", index=False)

# ------------------------------------------------------------
# 9. Which exact models improve most from 5 -> 20?
# ------------------------------------------------------------
improvers = traj.sort_values("Delta_RMSE_5_to_20").copy()
improvers.to_csv(OUT / "models_ranked_by_5_to_20_RMSE_change.csv", index=False)

# ------------------------------------------------------------
# 10. Concise report
# ------------------------------------------------------------
report = []
report.append("=" * 100)
report.append("ESOL QSARMIL — 288-MODEL STATISTICAL ANALYSIS")
report.append("=" * 100)
report.append("")
report.append("This analysis uses the existing 288 individual-model test predictions.")
report.append("No new QSARmil calculations were performed.")
report.append("The test-set metrics are descriptive and should not be treated as an")
report.append("independent final model-selection/validation procedure.")
report.append("")

report.append("RECORDS BY CONFORMER COUNT")
report.append(conformer.to_string(index=False))
report.append("")

report.append("DESCRIPTOR-LEVEL SUMMARY")
report.append(descriptor.to_string(index=False))
report.append("")

report.append("ARCHITECTURE-LEVEL SUMMARY")
report.append(architecture.to_string(index=False))
report.append("")

report.append("TOP 10 RECURRENCE")
report.append(recurrence.to_string(index=False))
report.append("")

report.append("5 -> 20 CONFORMER BEHAVIOR COUNTS")
report.append(behavior.to_string(index=False))
report.append("")

report.append("LARGEST 5 -> 20 RMSE IMPROVEMENTS")
cols = [
    "Descriptor", "Model", "RMSE_5conf", "RMSE_10conf",
    "RMSE_15conf", "RMSE_20conf", "Delta_RMSE_5_to_20",
    "R2_5conf", "R2_20conf"
]
report.append(improvers[cols].head(20).to_string(index=False))
report.append("")

report.append("LARGEST 5 -> 20 RMSE DETERIORATIONS")
report.append(improvers[cols].tail(20).sort_values(
    "Delta_RMSE_5_to_20", ascending=False
).to_string(index=False))
report.append("")

report.append("INTERPRETIVE NOTES")
report.append(
    "1. A negative Delta_RMSE means the 20-conformer model has lower RMSE than "
    "the corresponding 5-conformer model."
)
report.append(
    "2. A positive Delta_R2 means the 20-conformer model has higher R2."
)
report.append(
    "3. Repeated appearance in the top 10 indicates descriptive recurrence, "
    "not independent validation."
)
report.append(
    "4. Descriptor and architecture averages describe this ESOL experiment; "
    "they should not be generalized to other endpoints without replication."
)

(OUT / "ESOL_288_MODEL_STATISTICAL_ANALYSIS.txt").write_text(
    "\n".join(report), encoding="utf-8"
)

print("\n" + "=" * 100)
print("288-MODEL ANALYSIS COMPLETE")
print("=" * 100)
print(f"Output directory: {OUT}")

print("\nCONFORMER SUMMARY:")
print(conformer.to_string(index=False))

print("\nDESCRIPTOR SUMMARY:")
print(descriptor.to_string(index=False))

print("\nARCHITECTURE SUMMARY:")
print(architecture.to_string(index=False))

print("\nTOP-10 RECURRENCE:")
print(recurrence.to_string(index=False))

print("\n5 -> 20 BEHAVIOR:")
print(behavior.to_string(index=False))

print("\nTOP 10 EXACT MODELS BY 5 -> 20 RMSE IMPROVEMENT:")
print(improvers[cols].head(10).to_string(index=False))

print("\nReport:")
print(OUT / "ESOL_288_MODEL_STATISTICAL_ANALYSIS.txt")

