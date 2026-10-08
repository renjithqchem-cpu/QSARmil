import os
import numpy as np
import pandas as pd

OUT = "data/esol_3d_sil_hopt_vs_3d_mil/molecule_level"
os.makedirs(OUT, exist_ok=True)

sil = pd.read_csv("data/esol_3d_sil_neural_hopt/test_predictions.csv")

records = []

for nc in [5, 10, 15, 20]:
    mil = pd.read_csv(
        "data/esol_72model_audit/matched_test_predictions_"
        + str(nc) + "conf.csv"
    )

    m = sil.merge(mil, on="SMILES", suffixes=("_SIL", "_MIL"))
    y = m["Y_TRUE_ESOL_SIL"].values

    models = [
        x for x in sil.columns
        if x not in ["SMILES", "Y_TRUE_ESOL"] and x in mil.columns
    ]

    print(nc, "conf:", len(m), "molecules,", len(models), "models")

    for model in models:
        es = np.abs(y - m[model + "_SIL"].values)
        em = np.abs(y - m[model + "_MIL"].values)
        d = es - em

        for i in range(len(m)):
            records.append(
                [nc, m.iloc[i]["SMILES"], model, d[i]]
            )

df = pd.DataFrame(
    records,
    columns=["n_conformers", "SMILES", "model", "Delta_AE"]
)

df.to_csv(
    OUT + "/molecule_improvement_long.csv",
    index=False
)

summary = df.groupby(
    ["n_conformers", "SMILES"]
)["Delta_AE"].agg(
    ["mean", "median", "std", "min", "max"]
).reset_index()

summary.columns = [
    "n_conformers",
    "SMILES",
    "mean_Delta_AE",
    "median_Delta_AE",
    "SD_Delta_AE",
    "worst_Delta_AE",
    "best_Delta_AE"
]

counts = df.groupby(
    ["n_conformers", "SMILES"]
)["Delta_AE"].agg(
    models_improving=lambda x: (x > 0).sum(),
    models_worsening=lambda x: (x < 0).sum()
).reset_index()

summary = summary.merge(
    counts,
    on=["n_conformers", "SMILES"]
)

summary["fraction_improving"] = (
    summary["models_improving"] / 56.0
)

summary.to_csv(
    OUT + "/molecule_improvement_summary.csv",
    index=False
)

wide = summary.pivot(
    index="SMILES",
    columns="n_conformers",
    values="mean_Delta_AE"
)

wide.columns = [
    "DeltaAE_5",
    "DeltaAE_10",
    "DeltaAE_15",
    "DeltaAE_20"
]

wide = wide.reset_index()

dcols = [
    "DeltaAE_5",
    "DeltaAE_10",
    "DeltaAE_15",
    "DeltaAE_20"
]

wide["mean_all_counts"] = wide[dcols].mean(axis=1)
wide["n_positive_counts"] = (wide[dcols] > 0).sum(axis=1)
wide["n_negative_counts"] = (wide[dcols] < 0).sum(axis=1)

wide.to_csv(
    OUT + "/molecule_cross_ensemble_consistency.csv",
    index=False
)

print("")
print("SUMMARY")

for nc in [5, 10, 15, 20]:
    s = summary[summary["n_conformers"] == nc]

    print(
        nc,
        "conf | mean",
        round(s["mean_Delta_AE"].mean(), 6),
        "| median",
        round(s["mean_Delta_AE"].median(), 6),
        "| positive molecules",
        int((s["mean_Delta_AE"] > 0).sum()),
        "| negative molecules",
        int((s["mean_Delta_AE"] < 0).sum())
    )

print("")
print("MOST CONSISTENTLY HELPED")

print(
    wide.sort_values(
        ["n_positive_counts", "mean_all_counts"],
        ascending=[False, False]
    ).head(15).to_string(index=False)
)

print("")
print("MOST CONSISTENTLY HARMED")

print(
    wide.sort_values(
        ["n_negative_counts", "mean_all_counts"],
        ascending=[False, True]
    ).head(15).to_string(index=False)
)

print("")
print("DONE")
