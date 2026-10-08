from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest
from scipy.stats import bootstrap

BASE = Path.home() / "Documents" / "qsarmil"
INPUT = BASE / "data" / "esol_288_model_analysis" / "model_conformer_trajectory.csv"
OUT = BASE / "data" / "esol_conformer_effect_statistics"
OUT.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing {INPUT}\nRun esol_288_model_statistical_analysis.py first."
    )

df = pd.read_csv(INPUT)

required = [
    "Descriptor", "Model",
    "RMSE_5conf", "RMSE_20conf",
    "R2_5conf", "R2_20conf"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required).copy()

# 20 - 5 convention:
# negative ΔRMSE = improvement
# positive ΔR2 = improvement
df["Delta_RMSE"] = df["RMSE_20conf"] - df["RMSE_5conf"]
df["Delta_R2"] = df["R2_20conf"] - df["R2_5conf"]

rng = np.random.default_rng(20260917)

def bootstrap_mean_ci(values, n_boot=10000, seed_offset=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    r = np.random.default_rng(20260917 + seed_offset)
    samples = r.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def paired_stats(g, seed_offset=0):
    dr = g["Delta_RMSE"].to_numpy(float)
    d2 = g["Delta_R2"].to_numpy(float)

    n = len(dr)
    mean_dr = np.mean(dr)
    median_dr = np.median(dr)
    mean_d2 = np.mean(d2)
    median_d2 = np.median(d2)

    ci_dr = bootstrap_mean_ci(dr, seed_offset=seed_offset)
    ci_d2 = bootstrap_mean_ci(d2, seed_offset=seed_offset + 100000)

    nonzero = dr[np.abs(dr) > 1e-12]

    if len(nonzero) >= 2:
        try:
            w = wilcoxon(
                nonzero,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto"
            )
            wilcoxon_p = float(w.pvalue)
        except Exception:
            wilcoxon_p = np.nan

        npos = np.sum(nonzero > 0)
        nneg = np.sum(nonzero < 0)
        try:
            sign_p = float(
                binomtest(
                    min(npos, nneg),
                    n=npos + nneg,
                    p=0.5,
                    alternative="two-sided"
                ).pvalue
            )
        except Exception:
            sign_p = np.nan
    else:
        wilcoxon_p = np.nan
        sign_p = np.nan

    return {
        "N_models": n,
        "mean_Delta_RMSE_20_minus_5": mean_dr,
        "median_Delta_RMSE_20_minus_5": median_dr,
        "bootstrap95_low_Delta_RMSE": ci_dr[0],
        "bootstrap95_high_Delta_RMSE": ci_dr[1],
        "mean_Delta_R2_20_minus_5": mean_d2,
        "median_Delta_R2_20_minus_5": median_d2,
        "bootstrap95_low_Delta_R2": ci_d2[0],
        "bootstrap95_high_Delta_R2": ci_d2[1],
        "n_RMSE_improved": int(np.sum(dr < -1e-12)),
        "n_RMSE_worsened": int(np.sum(dr > 1e-12)),
        "n_RMSE_neutral": int(np.sum(np.abs(dr) <= 1e-12)),
        "Wilcoxon_p": wilcoxon_p,
        "Sign_test_p": sign_p,
    }

def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction; preserves NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    vals = p[mask]
    m = len(vals)
    if m == 0:
        return out

    order = np.argsort(vals)
    ranked = vals[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    restored = np.empty_like(adj)
    restored[order] = adj
    out[mask] = restored
    return out

# ------------------------------------------------------------
# Overall 72-model paired analysis
# ------------------------------------------------------------
overall = pd.DataFrame([paired_stats(df, 1)])
overall.to_csv(OUT / "overall_72_model_effect.csv", index=False)

# ------------------------------------------------------------
# Descriptor analysis
# ------------------------------------------------------------
desc_rows = []
for i, (name, g) in enumerate(df.groupby("Descriptor", sort=True)):
    x = paired_stats(g, 1000 + i)
    x["Descriptor"] = name
    desc_rows.append(x)

desc = pd.DataFrame(desc_rows)
desc["Wilcoxon_FDR"] = bh_fdr(desc["Wilcoxon_p"])
desc["Sign_test_FDR"] = bh_fdr(desc["Sign_test_p"])
desc = desc.sort_values("mean_Delta_RMSE_20_minus_5")
desc.to_csv(OUT / "descriptor_conformer_effect.csv", index=False)

# ------------------------------------------------------------
# Architecture analysis
# ------------------------------------------------------------
arch_rows = []
for i, (name, g) in enumerate(df.groupby("Model", sort=True)):
    x = paired_stats(g, 2000 + i)
    x["Model"] = name
    arch_rows.append(x)

arch = pd.DataFrame(arch_rows)
arch["Wilcoxon_FDR"] = bh_fdr(arch["Wilcoxon_p"])
arch["Sign_test_FDR"] = bh_fdr(arch["Sign_test_p"])
arch = arch.sort_values("mean_Delta_RMSE_20_minus_5")
arch.to_csv(OUT / "architecture_conformer_effect.csv", index=False)

# ------------------------------------------------------------
# Descriptor × architecture analysis
# ------------------------------------------------------------
combo_rows = []
for i, ((desc_name, model_name), g) in enumerate(
    df.groupby(["Descriptor", "Model"], sort=True)
):
    x = paired_stats(g, 3000 + i)
    x["Descriptor"] = desc_name
    x["Model"] = model_name
    combo_rows.append(x)

combo = pd.DataFrame(combo_rows)
combo["Wilcoxon_FDR"] = bh_fdr(combo["Wilcoxon_p"])
combo["Sign_test_FDR"] = bh_fdr(combo["Sign_test_p"])
combo = combo.sort_values("mean_Delta_RMSE_20_minus_5")
combo.to_csv(OUT / "descriptor_architecture_conformer_effect.csv", index=False)

# ------------------------------------------------------------
# Trajectory direction for all 72 models
# ------------------------------------------------------------
direction = []
for _, r in df.iterrows():
    dr = r["Delta_RMSE"]
    if dr < -0.01:
        label = "improved"
    elif dr > 0.01:
        label = "worsened"
    else:
        label = "approximately_stable"

    direction.append({
        "Descriptor": r["Descriptor"],
        "Model": r["Model"],
        "Model_full": r["Model_full"],
        "RMSE_5conf": r["RMSE_5conf"],
        "RMSE_10conf": r["RMSE_10conf"],
        "RMSE_15conf": r["RMSE_15conf"],
        "RMSE_20conf": r["RMSE_20conf"],
        "Delta_RMSE_20_minus_5": dr,
        "R2_5conf": r["R2_5conf"],
        "R2_20conf": r["R2_20conf"],
        "Delta_R2_20_minus_5": r["Delta_R2"],
        "behavior": label
    })

direction = pd.DataFrame(direction).sort_values(
    "Delta_RMSE_20_minus_5"
)
direction.to_csv(OUT / "all_72_conformer_effect_directions.csv", index=False)

# ------------------------------------------------------------
# Additional direct 5 -> 10 -> 15 -> 20 trajectory summaries
# ------------------------------------------------------------
traj_cols = [
    "Descriptor", "Model", "Model_full",
    "RMSE_5conf", "RMSE_10conf", "RMSE_15conf", "RMSE_20conf",
    "Delta_RMSE_5_to_10", "Delta_RMSE_10_to_15",
    "Delta_RMSE_15_to_20", "Delta_RMSE_5_to_20",
    "R2_5conf", "R2_10conf", "R2_15conf", "R2_20conf",
    "Delta_R2_5_to_10", "Delta_R2_10_to_15",
    "Delta_R2_15_to_20", "Delta_R2_5_to_20"
]
trajectory = df[[c for c in traj_cols if c in df.columns]].copy()
trajectory.to_csv(OUT / "full_model_trajectories.csv", index=False)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------
lines = []
lines.append("=" * 100)
lines.append("ESOL QSARMIL — RIGOROUS CONFORMER-EFFECT STATISTICS")
lines.append("=" * 100)
lines.append("")
lines.append(f"Input: {INPUT}")
lines.append(f"Exact descriptor × architecture models: {len(df)}")
lines.append("")
lines.append("SIGN CONVENTION")
lines.append("ΔRMSE = RMSE(20 conformers) - RMSE(5 conformers)")
lines.append("Negative ΔRMSE = improvement with 20 conformers.")
lines.append("Positive ΔR2 = improvement with 20 conformers.")
lines.append("")

lines.append("OVERALL 72-MODEL PAIRED RESULT")
lines.append(overall.to_string(index=False))
lines.append("")

lines.append("DESCRIPTOR-LEVEL EFFECT")
lines.append(desc.to_string(index=False))
lines.append("")

lines.append("ARCHITECTURE-LEVEL EFFECT")
lines.append(arch.to_string(index=False))
lines.append("")

lines.append("DESCRIPTOR × ARCHITECTURE EFFECT")
lines.append(combo.to_string(index=False))
lines.append("")

lines.append("5 -> 20 DIRECTION COUNTS")
lines.append(
    direction["behavior"].value_counts()
    .rename_axis("behavior")
    .to_string()
)
lines.append("")

lines.append("LARGEST RMSE IMPROVEMENTS (5 -> 20)")
lines.append(
    direction[
        [
            "Descriptor", "Model",
            "RMSE_5conf", "RMSE_10conf", "RMSE_15conf", "RMSE_20conf",
            "Delta_RMSE_20_minus_5",
            "R2_5conf", "R2_20conf", "Delta_R2_20_minus_5"
        ]
    ].head(20).to_string(index=False)
)
lines.append("")

lines.append("LARGEST RMSE DETERIORATIONS (5 -> 20)")
lines.append(
    direction[
        [
            "Descriptor", "Model",
            "RMSE_5conf", "RMSE_10conf", "RMSE_15conf", "RMSE_20conf",
            "Delta_RMSE_20_minus_5",
            "R2_5conf", "R2_20conf", "Delta_R2_20_minus_5"
        ]
    ].tail(20).sort_values(
        "Delta_RMSE_20_minus_5", ascending=False
    ).to_string(index=False)
)
lines.append("")

lines.append("INTERPRETATION RULES")
lines.append(
    "These tests compare model-level performance across conformer settings. "
    "They do not test molecule-level paired prediction errors and should not "
    "be interpreted as independent molecular replicates."
)
lines.append(
    "FDR correction is provided for the descriptor, architecture and "
    "descriptor×architecture exploratory families."
)
lines.append(
    "A small p-value alone is not treated as evidence of a practically "
    "important effect; effect size and bootstrap uncertainty must also be considered."
)
lines.append(
    "The analysis is restricted to ESOL and the existing QSARmil workflow."
)

(OUT / "ESOL_CONFORMER_EFFECT_STATISTICS_REPORT.txt").write_text(
    "\n".join(lines), encoding="utf-8"
)

print("\n" + "=" * 100)
print("RIGOROUS CONFORMER-EFFECT ANALYSIS COMPLETE")
print("=" * 100)

print("\nOVERALL 72-MODEL RESULT:")
print(overall.to_string(index=False))

print("\nDESCRIPTOR EFFECT:")
print(desc.to_string(index=False))

print("\nARCHITECTURE EFFECT:")
print(arch.to_string(index=False))

print("\nTOP DESCRIPTOR × ARCHITECTURE EFFECTS:")
print(combo.head(20).to_string(index=False))

print("\n5 -> 20 DIRECTION COUNTS:")
print(direction["behavior"].value_counts())

print("\nLargest improvements:")
print(direction[[
    "Descriptor","Model","Delta_RMSE_20_minus_5","Delta_R2_20_minus_5"
]].head(10).to_string(index=False))

print("\nOutput:")
print(OUT)

