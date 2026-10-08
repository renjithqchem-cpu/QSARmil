from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path.home() / "Documents" / "qsarmil"
INPUT = BASE / "data" / "esol_analysis" / "prediction_changes.csv"
OUT = BASE / "data" / "esol_flexibility_effect_size_analysis"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
BOOTSTRAP_N = 10000
PERMUTATION_N = 10000
rng = np.random.default_rng(SEED)

df = pd.read_csv(INPUT)

required = [
    "conformers", "target", "rotatable_bonds", "flexibility",
    "SIL_prediction", "MIL_prediction",
    "prediction_change_MIL_minus_SIL",
    "abs_prediction_change", "abs_error_SIL", "abs_error_MIL",
    "error_improvement"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["RotB"] = pd.to_numeric(df["rotatable_bonds"], errors="coerce")
df["error_improvement"] = pd.to_numeric(df["error_improvement"], errors="coerce")
df["abs_prediction_change"] = pd.to_numeric(df["abs_prediction_change"], errors="coerce")
df = df.dropna(subset=["conformers", "RotB", "error_improvement", "abs_prediction_change"]).copy()

# ---------- statistics helpers ----------

def bootstrap_ci(x, statistic=np.mean, n=BOOTSTRAP_N, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = statistic(x[idx], axis=1)
    lo, hi = np.quantile(vals, [alpha/2, 1-alpha/2])
    return float(statistic(x)), float(lo), float(hi)

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    return np.nan if pooled == 0 else float((np.mean(a)-np.mean(b))/pooled)

def rank_biserial_from_mannwhitney(a, b):
    # Positive value means values in a tend to be larger than b.
    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    return float(2*u/(n1*n2) - 1)

def permutation_mean_diff(a, b, n=PERMUTATION_N):
    a, b = np.asarray(a, float), np.asarray(b, float)
    observed = np.mean(a) - np.mean(b)
    pooled = np.concatenate([a, b])
    n_a = len(a)
    diffs = np.empty(n)
    for i in range(n):
        perm = rng.permutation(pooled)
        diffs[i] = np.mean(perm[:n_a]) - np.mean(perm[n_a:])
    p = (np.sum(np.abs(diffs) >= abs(observed)) + 1) / (n + 1)
    return float(observed), float(p)

# ---------- 1. subgroup effect sizes ----------

rows = []
for conf in sorted(df["conformers"].unique()):
    sub = df[df["conformers"] == conf].copy()
    groups = {
        "All": sub,
        "RotB<4": sub[sub["RotB"] < 4],
        "RotB>=4": sub[sub["RotB"] >= 4],
        "RotB>=7": sub[sub["RotB"] >= 7],
    }

    for name, g in groups.items():
        x = g["error_improvement"].to_numpy()
        mean, mean_lo, mean_hi = bootstrap_ci(x, np.mean)
        med, med_lo, med_hi = bootstrap_ci(x, np.median)
        rows.append({
            "Conformers": int(conf),
            "Group": name,
            "N": len(g),
            "Mean_error_improvement": mean,
            "Mean_CI95_low": mean_lo,
            "Mean_CI95_high": mean_hi,
            "Median_error_improvement": med,
            "Median_CI95_low": med_lo,
            "Median_CI95_high": med_hi,
            "Fraction_MIL_improved": float(np.mean(x > 0)) if len(x) else np.nan,
            "Fraction_MIL_worsened": float(np.mean(x < 0)) if len(x) else np.nan,
        })

effect_summary = pd.DataFrame(rows)
effect_summary.to_csv(OUT / "bootstrap_effect_summary.csv", index=False)

# ---------- 2. flexible vs non-flexible comparison ----------

cmp_rows = []
for conf in sorted(df["conformers"].unique()):
    sub = df[df["conformers"] == conf]
    less = sub.loc[sub["RotB"] < 4, "error_improvement"].to_numpy()
    flex = sub.loc[sub["RotB"] >= 4, "error_improvement"].to_numpy()

    diff, p_perm = permutation_mean_diff(flex, less)

    # Bootstrap CI for difference of means.
    boot = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        bf = flex[rng.integers(0, len(flex), len(flex))]
        bl = less[rng.integers(0, len(less), len(less))]
        boot[i] = np.mean(bf) - np.mean(bl)
    ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])

    t_stat, t_p = stats.ttest_ind(flex, less, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(flex, less, alternative="two-sided")
    d = cohens_d(flex, less)
    rbc = rank_biserial_from_mannwhitney(flex, less)

    cmp_rows.append({
        "Conformers": int(conf),
        "N_RotB_lt4": len(less),
        "N_RotB_ge4": len(flex),
        "Mean_RotB_lt4": np.mean(sub.loc[sub["RotB"] < 4, "RotB"]),
        "Mean_RotB_ge4": np.mean(sub.loc[sub["RotB"] >= 4, "RotB"]),
        "Mean_improvement_lt4": np.mean(less),
        "Mean_improvement_ge4": np.mean(flex),
        "Mean_difference_ge4_minus_lt4": diff,
        "Difference_CI95_low": ci_lo,
        "Difference_CI95_high": ci_hi,
        "Permutation_p": p_perm,
        "Welch_t_p": t_p,
        "MannWhitney_p": u_p,
        "Cohens_d_ge4_vs_lt4": d,
        "Rank_biserial_ge4_vs_lt4": rbc,
    })

comparison = pd.DataFrame(cmp_rows)
comparison.to_csv(OUT / "flexible_vs_less_flexible_effects.csv", index=False)

# ---------- 3. >=7 subgroup comparison against <7 ----------

rows7 = []
for conf in sorted(df["conformers"].unique()):
    sub = df[df["conformers"] == conf]
    hi = sub.loc[sub["RotB"] >= 7, "error_improvement"].to_numpy()
    lo = sub.loc[sub["RotB"] < 7, "error_improvement"].to_numpy()

    diff, p_perm = permutation_mean_diff(hi, lo)
    boot = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        bh = hi[rng.integers(0, len(hi), len(hi))]
        bl = lo[rng.integers(0, len(lo), len(lo))]
        boot[i] = np.mean(bh) - np.mean(bl)
    ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])

    rows7.append({
        "Conformers": int(conf),
        "N_RotB_ge7": len(hi),
        "N_RotB_lt7": len(lo),
        "Mean_improvement_ge7": np.mean(hi),
        "Mean_improvement_lt7": np.mean(lo),
        "Difference_ge7_minus_lt7": diff,
        "Difference_CI95_low": ci_lo,
        "Difference_CI95_high": ci_hi,
        "Permutation_p": p_perm,
        "Cohens_d_ge7_vs_lt7": cohens_d(hi, lo),
        "Rank_biserial_ge7_vs_lt7": rank_biserial_from_mannwhitney(hi, lo),
    })

comparison7 = pd.DataFrame(rows7)
comparison7.to_csv(OUT / "high_flexibility_effects.csv", index=False)

# ---------- 4. continuous RotB correlations ----------

corr_rows = []
for conf in sorted(df["conformers"].unique()):
    sub = df[df["conformers"] == conf]
    for variable in ["error_improvement", "abs_prediction_change"]:
        x = sub["RotB"].to_numpy()
        y = sub[variable].to_numpy()
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        corr_rows.append({
            "Conformers": int(conf),
            "Variable": variable,
            "N": len(sub),
            "Pearson_r": pr,
            "Pearson_p": pp,
            "Spearman_rho": sr,
            "Spearman_p": sp,
        })
corr = pd.DataFrame(corr_rows)
corr.to_csv(OUT / "continuous_RotB_correlations.csv", index=False)

# ---------- 5. Molecule-level counts / direction ----------

summary_rows = []
for conf in sorted(df["conformers"].unique()):
    sub = df[df["conformers"] == conf]
    for label, mask in [
        ("All", np.ones(len(sub), dtype=bool)),
        ("RotB<4", sub["RotB"].to_numpy() < 4),
        ("RotB>=4", sub["RotB"].to_numpy() >= 4),
        ("RotB>=7", sub["RotB"].to_numpy() >= 7),
    ]:
        x = sub.loc[mask, "error_improvement"].to_numpy()
        summary_rows.append({
            "Conformers": int(conf),
            "Group": label,
            "N": len(x),
            "Improved": int(np.sum(x > 0)),
            "Worsened": int(np.sum(x < 0)),
            "Equal": int(np.sum(x == 0)),
            "Improved_fraction": float(np.mean(x > 0)) if len(x) else np.nan,
            "Worsened_fraction": float(np.mean(x < 0)) if len(x) else np.nan,
            "Mean_abs_error_improvement": float(np.mean(x)) if len(x) else np.nan,
            "Median_abs_error_improvement": float(np.median(x)) if len(x) else np.nan,
        })
direction = pd.DataFrame(summary_rows)
direction.to_csv(OUT / "direction_summary.csv", index=False)

# ---------- figures ----------

# Figure 1: mean improvement with bootstrap CI
plt.figure(figsize=(9, 6))
for group in ["RotB<4", "RotB>=4", "RotB>=7"]:
    g = effect_summary[effect_summary["Group"] == group].sort_values("Conformers")
    plt.errorbar(
        g["Conformers"],
        g["Mean_error_improvement"],
        yerr=[
            g["Mean_error_improvement"] - g["Mean_CI95_low"],
            g["Mean_CI95_high"] - g["Mean_error_improvement"]
        ],
        marker="o",
        capsize=4,
        label=group
    )
plt.axhline(0, linewidth=1)
plt.xlabel("Number of conformers")
plt.ylabel("Mean error improvement (SIL absolute error − MIL absolute error)")
plt.title("Flexibility-stratified prediction benefit")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "mean_error_improvement_bootstrap_CI.png", dpi=300)
plt.close()

# Figure 2: difference between flexible and less-flexible groups
g = comparison.sort_values("Conformers")
plt.figure(figsize=(9, 6))
plt.errorbar(
    g["Conformers"],
    g["Mean_difference_ge4_minus_lt4"],
    yerr=[
        g["Mean_difference_ge4_minus_lt4"] - g["Difference_CI95_low"],
        g["Difference_CI95_high"] - g["Mean_difference_ge4_minus_lt4"]
    ],
    marker="o",
    capsize=4
)
plt.axhline(0, linewidth=1)
plt.xlabel("Number of conformers")
plt.ylabel("Difference in mean error improvement (RotB≥4 − RotB<4)")
plt.title("Does flexibility modify the benefit of conformer MIL?")
plt.tight_layout()
plt.savefig(OUT / "flexibility_effect_difference.png", dpi=300)
plt.close()

# Figure 3: continuous relationship, all conformers
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, conf in zip(axes.ravel(), sorted(df["conformers"].unique())):
    sub = df[df["conformers"] == conf]
    ax.scatter(sub["RotB"], sub["error_improvement"], alpha=0.55)
    ax.axhline(0, linewidth=1)
    ax.set_title(f"{int(conf)} conformers")
    ax.set_xlabel("Rotatable bonds")
    ax.set_ylabel("Error improvement")
fig.suptitle("Molecule-level error improvement vs molecular flexibility", y=1.01)
fig.tight_layout()
fig.savefig(OUT / "error_improvement_vs_RotB_all_conformers.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 4: flexible vs less-flexible distributions
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, conf in zip(axes.ravel(), sorted(df["conformers"].unique())):
    sub = df[df["conformers"] == conf]
    a = sub.loc[sub["RotB"] < 4, "error_improvement"]
    b = sub.loc[sub["RotB"] >= 4, "error_improvement"]
    ax.boxplot([a, b], labels=["RotB<4", "RotB≥4"], showmeans=True)
    ax.axhline(0, linewidth=1)
    ax.set_title(f"{int(conf)} conformers")
    ax.set_ylabel("Error improvement")
fig.suptitle("Distribution of molecule-level prediction benefit", y=1.01)
fig.tight_layout()
fig.savefig(OUT / "error_improvement_distributions.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- report ----------

lines = []
lines.append("=" * 80)
lines.append("ESOL FLEXIBILITY-STRATIFIED EFFECT-SIZE ANALYSIS")
lines.append("=" * 80)
lines.append(f"Input: {INPUT}")
lines.append(f"Output: {OUT}")
lines.append(f"Bootstrap replicates: {BOOTSTRAP_N}")
lines.append(f"Permutation replicates: {PERMUTATION_N}")
lines.append("")

lines.append("BOOTSTRAP EFFECT SUMMARY")
lines.append(effect_summary.to_string(index=False))
lines.append("")

lines.append("ROT B >=4 VS ROT B <4")
lines.append(comparison.to_string(index=False))
lines.append("")

lines.append("ROT B >=7 VS ROT B <7")
lines.append(comparison7.to_string(index=False))
lines.append("")

lines.append("CONTINUOUS ROTB CORRELATIONS")
lines.append(corr.to_string(index=False))
lines.append("")

lines.append("DIRECTION SUMMARY")
lines.append(direction.to_string(index=False))
lines.append("")

lines.append("INTERPRETATION GUIDE")
lines.append("- Positive error improvement means MIL has lower absolute prediction error than SIL.")
lines.append("- The key flexibility test is whether the flexible-vs-less-flexible mean difference has a CI excluding zero and/or permutation p < 0.05.")
lines.append("- Cohen's d and rank-biserial effect sizes quantify magnitude; they are not evidence of statistical significance by themselves.")
lines.append("- RotB is a proxy for flexibility. It does not directly measure conformational diversity.")
lines.append("- The >=7 RotB subgroup is small and should be treated cautiously.")
lines.append("- These analyses use the existing ESOL split and therefore require replication on additional splits/seeds before strong generalization claims.")
(OUT / "ESOL_FLEXIBILITY_EFFECT_SIZE_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")

print("=" * 80)
print("ESOL FLEXIBILITY EFFECT-SIZE ANALYSIS COMPLETE")
print("=" * 80)
print(f"Output directory: {OUT}")
print("\nBOOTSTRAP EFFECT SUMMARY")
print(effect_summary.to_string(index=False))
print("\nROT B >=4 VS ROT B <4")
print(comparison.to_string(index=False))
print("\nROT B >=7 VS ROT B <7")
print(comparison7.to_string(index=False))
print("\nCONTINUOUS ROTB CORRELATIONS")
print(corr.to_string(index=False))
print("\nReport:", OUT / "ESOL_FLEXIBILITY_EFFECT_SIZE_REPORT.txt")
