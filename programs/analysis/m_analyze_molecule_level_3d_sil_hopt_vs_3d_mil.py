import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

SIL = "data/esol_3d_sil_neural_hopt/test_predictions.csv"
OUT = "data/esol_3d_sil_hopt_vs_3d_mil/molecule_level"

os.makedirs(OUT, exist_ok=True)

sil = pd.read_csv(SIL)

print("=" * 70)
print("MOLECULE-LEVEL 3D-SIL vs 3D-MIL ANALYSIS")
print("=" * 70)
print("3D-SIL rows:", len(sil))
print("3D-SIL prediction columns:", len(sil.columns) - 2)

all_results = []

for nconf in [5, 10, 15, 20]:

    mil_file = (
        "data/esol_72model_audit/"
        f"matched_test_predictions_{nconf}conf.csv"
    )

    mil = pd.read_csv(mil_file)

    merged = sil.merge(
        mil,
        on="SMILES",
        how="inner",
        suffixes=("_SIL", "_MIL")
    )

    target = merged["Y_TRUE_ESOL_SIL"]

    models = [
        c for c in sil.columns
        if c not in ["SMILES", "Y_TRUE_ESOL"]
        and c in mil.columns
    ]

    print()
    print("Conformers:", nconf)
    print("Matched molecules:", len(merged))
    print("Matched models:", len(models))

    for model in models:

        y = target.to_numpy(dtype=float)

        p_sil = merged[model + "_SIL"].to_numpy(dtype=float)
        p_mil = merged[model + "_MIL"].to_numpy(dtype=float)

        ae_sil = np.abs(y - p_sil)
        ae_mil = np.abs(y - p_mil)

        delta = ae_sil - ae_mil

        w = wilcoxon(
            delta,
            zero_method="wilcox",
            alternative="two-sided"
        )

        positive = int(np.sum(delta > 0))
        negative = int(np.sum(delta < 0))

        sign = binomtest(
            positive,
            positive + negative,
            p=0.5
        )

        all_results.append({
            "n_conformers": nconf,
            "model": model,
            "mean_AE_SIL": np.mean(ae_sil),
            "mean_AE_MIL": np.mean(ae_mil),
            "mean_Delta_AE": np.mean(delta),
            "median_Delta_AE": np.median(delta),
            "SD_Delta_AE": np.std(delta, ddof=1),
            "min_Delta_AE": np.min(delta),
            "max_Delta_AE": np.max(delta),
            "molecules_improved": positive,
            "molecules_worsened": negative,
            "fraction_improved": positive / len(delta),
            "Wilcoxon_p": w.pvalue,
            "sign_test_p": sign.pvalue
        })

results = pd.DataFrame(all_results)

outfile = OUT + "/molecule_level_model_statistics.csv"
results.to_csv(outfile, index=False)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

summary = (
    results
    .groupby("n_conformers")
    .agg(
        models=("model", "count"),
        mean_model_Delta_AE=("mean_Delta_AE", "mean"),
        median_model_Delta_AE=("mean_Delta_AE", "median"),
        models_with_positive_Delta=("mean_Delta_AE", lambda x: np.sum(x > 0)),
        models_with_negative_Delta=("mean_Delta_AE", lambda x: np.sum(x < 0))
    )
    .reset_index()
)

print(summary.to_string(index=False))

print()
print("Saved:", outfile)
print("DONE")
