import numpy as np
import pandas as pd

f = pd.read_csv(
    "data/esol_3d_sil_hopt_vs_3d_mil/paired_model_comparison.csv"
)

d = f.pivot(
    index="model",
    columns="n_conformers",
    values="Delta_RMSE_SIL_minus_MIL"
)

d.columns = ["D5", "D10", "D15", "D20"]

d["mean_Delta"] = d[["D5", "D10", "D15", "D20"]].mean(axis=1)

d["n_positive"] = (d[["D5", "D10", "D15", "D20"]] > 0).sum(axis=1)

d["n_negative"] = (d[["D5", "D10", "D15", "D20"]] < 0).sum(axis=1)

d["all_positive"] = d["n_positive"] == 4

d["all_negative"] = d["n_negative"] == 4

d["range"] = d[["D5", "D10", "D15", "D20"]].max(axis=1) - d[
    ["D5", "D10", "D15", "D20"]
].min(axis=1)

d["best_count"] = d[["D5", "D10", "D15", "D20"]].idxmax(axis=1)

d["worst_count"] = d[["D5", "D10", "D15", "D20"]].idxmin(axis=1)

d["monotonic_increasing"] = (
    (d["D10"] >= d["D5"]) &
    (d["D15"] >= d["D10"]) &
    (d["D20"] >= d["D15"])
)

d["monotonic_decreasing"] = (
    (d["D10"] <= d["D5"]) &
    (d["D15"] <= d["D10"]) &
    (d["D20"] <= d["D15"])
)

out = (
    "data/esol_3d_sil_hopt_vs_3d_mil/"
    "model_conformer_trajectories.csv"
)

d.reset_index().to_csv(out, index=False)

print("")
print("=" * 70)
print("MODEL CONFORMER TRAJECTORIES")
print("=" * 70)

print(d.reset_index().to_string(index=False))

print("")
print("=" * 70)
print("TRAJECTORY COUNTS")
print("=" * 70)

print("Total models:", len(d))
print("All four counts positive:", int(d["all_positive"].sum()))
print("All four counts negative:", int(d["all_negative"].sum()))
print("Mixed:", int((~d["all_positive"] & ~d["all_negative"]).sum()))
print("Monotonic increasing:", int(d["monotonic_increasing"].sum()))
print("Monotonic decreasing:", int(d["monotonic_decreasing"].sum()))

print("")
print("=" * 70)
print("BEST CONFORMER COUNT")
print("=" * 70)

print(d["best_count"].value_counts().sort_index())

print("")
print("=" * 70)
print("LARGEST POSITIVE MEAN EFFECTS")
print("=" * 70)

print(
    d.sort_values("mean_Delta", ascending=False)
    .head(15)
    .to_string()
)

print("")
print("=" * 70)
print("LARGEST NEGATIVE MEAN EFFECTS")
print("=" * 70)

print(
    d.sort_values("mean_Delta")
    .head(15)
    .to_string()
)

print("")
print("OUTPUT:")
print(out)
print("DONE")
