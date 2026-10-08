import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

INFILE = "data/esol_3d_sil_vs_3d_mil/paired_model_comparison.csv"
OUTDIR = "data/esol_3d_sil_vs_3d_mil/statistical_analysis"
os.makedirs(OUTDIR, exist_ok=True)

RNG = np.random.default_rng(42)
N_BOOT = 20000


def bootstrap_mean_ci(x, rng, n_boot=N_BOOT, alpha=0.05):
    x = np.asarray(x, dtype=float)
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True)
    means = samples.mean(axis=1)
    return np.quantile(means, alpha/2), np.quantile(means, 1-alpha/2)


def bootstrap_median_ci(x, rng, n_boot=N_BOOT, alpha=0.05):
    x = np.asarray(x, dtype=float)
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True)
    meds = np.median(samples, axis=1)
    return np.quantile(meds, alpha/2), np.quantile(meds, 1-alpha/2)


def cohens_d_paired(x):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return np.nan if sd == 0 else x.mean() / sd


def safe_wilcoxon(x):
    x = np.asarray(x, dtype=float)
    if np.allclose(x, 0):
        return 0.0, 1.0
    stat, p = wilcoxon(x, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(stat), float(p)


def fdr_bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


print("=" * 90)
print("STATISTICAL ANALYSIS: 3D-SIL vs 3D-MIL")
print("=" * 90)
print(f"Input: {INFILE}")

if not os.path.exists(INFILE):
    raise FileNotFoundError(INFILE)

df = pd.read_csv(INFILE)

required = [
    "n_conformers", "model", "SIL_RMSE", "MIL_RMSE",
    "SIL_MAE", "MIL_MAE", "Delta_RMSE_SIL_minus_MIL",
    "Delta_MAE_SIL_minus_MIL", "MIL_improves_RMSE", "MIL_improves_MAE"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

print(f"Rows: {len(df)}")
print(f"Unique models: {df.model.nunique()}")
print(f"Conformer counts: {sorted(df.n_conformers.unique())}")

# The paired comparison file stores model names as:
# DESCRIPTOR|ARCHITECTURE
# Reconstruct these two factors for the descriptor- and
# architecture-level analyses.
if "descriptor" not in df.columns:
    df["descriptor"] = (
        df["model"].astype(str)
        .str.split("|", n=1)
        .str[0]
    )

if "architecture" not in df.columns:
    df["architecture"] = (
        df["model"].astype(str)
        .str.split("|", n=1)
        .str[1]
    )

print(f"Descriptors: {df.descriptor.nunique()}")
print(f"Architectures: {df.architecture.nunique()}")

# -------------------------------------------------------------------------
# 1. Paired statistics by conformer count
# -------------------------------------------------------------------------
rows = []
for nconf in sorted(df.n_conformers.unique()):
    sub = df[df.n_conformers == nconf].copy()
    d_rmse = sub["Delta_RMSE_SIL_minus_MIL"].to_numpy(float)
    d_mae = sub["Delta_MAE_SIL_minus_MIL"].to_numpy(float)

    rmse_ci = bootstrap_mean_ci(d_rmse, RNG)
    rmse_med_ci = bootstrap_median_ci(d_rmse, RNG)
    mae_ci = bootstrap_mean_ci(d_mae, RNG)
    stat_rmse, p_rmse = safe_wilcoxon(d_rmse)
    stat_mae, p_mae = safe_wilcoxon(d_mae)

    improved_rmse = int((d_rmse > 0).sum())
    improved_mae = int((d_mae > 0).sum())
    n = len(sub)

    bt_rmse = binomtest(improved_rmse, n=n, p=0.5, alternative="two-sided")
    bt_mae = binomtest(improved_mae, n=n, p=0.5, alternative="two-sided")

    rows.append({
        "n_conformers": nconf,
        "n_models": n,
        "mean_delta_RMSE": d_rmse.mean(),
        "mean_delta_RMSE_CI95_low": rmse_ci[0],
        "mean_delta_RMSE_CI95_high": rmse_ci[1],
        "median_delta_RMSE": np.median(d_rmse),
        "median_delta_RMSE_CI95_low": rmse_med_ci[0],
        "median_delta_RMSE_CI95_high": rmse_med_ci[1],
        "SD_delta_RMSE": d_rmse.std(ddof=1),
        "Cohens_d_paired_RMSE": cohens_d_paired(d_rmse),
        "Wilcoxon_stat_RMSE": stat_rmse,
        "Wilcoxon_p_RMSE": p_rmse,
        "models_improved_RMSE": improved_rmse,
        "fraction_improved_RMSE": improved_rmse / n,
        "binomial_p_RMSE": bt_rmse.pvalue,
        "mean_delta_MAE": d_mae.mean(),
        "mean_delta_MAE_CI95_low": mae_ci[0],
        "mean_delta_MAE_CI95_high": mae_ci[1],
        "median_delta_MAE": np.median(d_mae),
        "SD_delta_MAE": d_mae.std(ddof=1),
        "Cohens_d_paired_MAE": cohens_d_paired(d_mae),
        "Wilcoxon_stat_MAE": stat_mae,
        "Wilcoxon_p_MAE": p_mae,
        "models_improved_MAE": improved_mae,
        "fraction_improved_MAE": improved_mae / n,
        "binomial_p_MAE": bt_mae.pvalue,
    })

paired_summary = pd.DataFrame(rows)
paired_summary.to_csv(f"{OUTDIR}/paired_statistics_by_conformer_count.csv", index=False)

print("\nPAIRED STATISTICS")
print(paired_summary[[
    "n_conformers", "n_models", "mean_delta_RMSE",
    "mean_delta_RMSE_CI95_low", "mean_delta_RMSE_CI95_high",
    "median_delta_RMSE", "Cohens_d_paired_RMSE",
    "Wilcoxon_p_RMSE", "models_improved_RMSE",
    "fraction_improved_RMSE", "binomial_p_RMSE"
]].to_string(index=False))

# -------------------------------------------------------------------------
# 2. Multiple-testing correction across the 4 conformer-count tests
# -------------------------------------------------------------------------
paired_summary["Wilcoxon_q_RMSE"] = fdr_bh(paired_summary["Wilcoxon_p_RMSE"])
paired_summary["binomial_q_RMSE"] = fdr_bh(paired_summary["binomial_p_RMSE"])
paired_summary.to_csv(f"{OUTDIR}/paired_statistics_by_conformer_count.csv", index=False)

# -------------------------------------------------------------------------
# 3. Descriptor-level effects at each conformer count
# -------------------------------------------------------------------------
desc_rows = []
for (nconf, desc), sub in df.groupby(["n_conformers", "descriptor"], dropna=False):
    x = sub["Delta_RMSE_SIL_minus_MIL"].to_numpy(float)
    ci = bootstrap_mean_ci(x, RNG)
    stat, p = safe_wilcoxon(x)
    desc_rows.append({
        "n_conformers": nconf,
        "descriptor": desc,
        "n_models": len(x),
        "mean_delta_RMSE": x.mean(),
        "mean_delta_RMSE_CI95_low": ci[0],
        "mean_delta_RMSE_CI95_high": ci[1],
        "median_delta_RMSE": np.median(x),
        "SD_delta_RMSE": x.std(ddof=1),
        "Cohens_d_paired": cohens_d_paired(x),
        "Wilcoxon_p": p,
        "models_improved": int((x > 0).sum()),
        "fraction_improved": float((x > 0).mean()),
    })

desc_stats = pd.DataFrame(desc_rows)
desc_stats["Wilcoxon_q"] = fdr_bh(desc_stats["Wilcoxon_p"])
desc_stats.to_csv(f"{OUTDIR}/descriptor_statistics.csv", index=False)

# -------------------------------------------------------------------------
# 4. Architecture-level effects at each conformer count
# -------------------------------------------------------------------------
arch_rows = []
for (nconf, arch), sub in df.groupby(["n_conformers", "architecture"], dropna=False):
    x = sub["Delta_RMSE_SIL_minus_MIL"].to_numpy(float)
    ci = bootstrap_mean_ci(x, RNG)
    stat, p = safe_wilcoxon(x)
    arch_rows.append({
        "n_conformers": nconf,
        "architecture": arch,
        "n_models": len(x),
        "mean_delta_RMSE": x.mean(),
        "mean_delta_RMSE_CI95_low": ci[0],
        "mean_delta_RMSE_CI95_high": ci[1],
        "median_delta_RMSE": np.median(x),
        "SD_delta_RMSE": x.std(ddof=1),
        "Cohens_d_paired": cohens_d_paired(x),
        "Wilcoxon_p": p,
        "models_improved": int((x > 0).sum()),
        "fraction_improved": float((x > 0).mean()),
    })

arch_stats = pd.DataFrame(arch_rows)
arch_stats["Wilcoxon_q"] = fdr_bh(arch_stats["Wilcoxon_p"])
arch_stats.to_csv(f"{OUTDIR}/architecture_statistics.csv", index=False)

# -------------------------------------------------------------------------
# 5. Paired trajectories: 5 -> 10 -> 15 -> 20 for each model
# -------------------------------------------------------------------------
pivot = df.pivot(index="model", columns="n_conformers", values="Delta_RMSE_SIL_minus_MIL")
pivot.columns = [f"Delta_RMSE_{int(c)}conf" for c in pivot.columns]
pivot.reset_index().to_csv(f"{OUTDIR}/model_delta_rmse_trajectory.csv", index=False)

# -------------------------------------------------------------------------
# 6. Rank models at each conformer count and stability of benefit
# -------------------------------------------------------------------------
rank_rows = []
for nconf in sorted(df.n_conformers.unique()):
    sub = df[df.n_conformers == nconf].sort_values("Delta_RMSE_SIL_minus_MIL", ascending=False)
    for rank, (_, row) in enumerate(sub.iterrows(), start=1):
        rank_rows.append({
            "n_conformers": nconf,
            "rank": rank,
            "model": row["model"],
            "delta_RMSE": row["Delta_RMSE_SIL_minus_MIL"],
        })

pd.DataFrame(rank_rows).to_csv(f"{OUTDIR}/model_rankings.csv", index=False)

# -------------------------------------------------------------------------
# 7. Models consistently improved at all four conformer counts
# -------------------------------------------------------------------------
wide_imp = df.pivot(index="model", columns="n_conformers", values="MIL_improves_RMSE")
wide_imp.columns = [f"improved_{int(c)}conf" for c in wide_imp.columns]
wide_imp["improved_all_4"] = wide_imp.all(axis=1)
wide_imp["improved_3_or_more"] = wide_imp.sum(axis=1) >= 3
wide_imp.reset_index().to_csv(f"{OUTDIR}/model_improvement_consistency.csv", index=False)

# -------------------------------------------------------------------------
# 8. 5 -> 20 change in ensemble benefit
# -------------------------------------------------------------------------
wide_delta = df.pivot(index="model", columns="n_conformers", values="Delta_RMSE_SIL_minus_MIL")
wide_delta["change_20_minus_5"] = wide_delta[20] - wide_delta[5]
wide_delta["improvement_gain_20_vs_5"] = -wide_delta["change_20_minus_5"]
wide_delta.reset_index().to_csv(f"{OUTDIR}/model_5_to_20_change.csv", index=False)

print("\n" + "=" * 90)
print("TOP 15 MODELS BY MEAN ΔRMSE ACROSS 5/10/15/20")
print("=" * 90)
mean_model = (
    df.groupby("model")
      .agg(
          mean_delta_RMSE=("Delta_RMSE_SIL_minus_MIL", "mean"),
          median_delta_RMSE=("Delta_RMSE_SIL_minus_MIL", "median"),
          n_improved=("MIL_improves_RMSE", "sum"),
      )
      .sort_values("mean_delta_RMSE", ascending=False)
)
print(mean_model.head(15).to_string())

print("\n" + "=" * 90)
print("BOTTOM 15 MODELS BY MEAN ΔRMSE ACROSS 5/10/15/20")
print("=" * 90)
print(mean_model.tail(15).sort_values("mean_delta_RMSE").to_string())

print("\nOutput directory:")
print(OUTDIR)
print("Files:")
for f in sorted(os.listdir(OUTDIR)):
    print(" ", f)
print("\nDONE")

