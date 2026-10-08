from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, wilcoxon

BASE = Path.home() / "Documents" / "qsarmil" / "data"
ANALYSIS = BASE / "esol_analysis"
OUT = BASE / "esol_flexibility_analysis"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CHANGE_FILE = ANALYSIS / "prediction_changes.csv"

# ---------------------------------------------------------------------
# Load existing molecule-level prediction comparison
# ---------------------------------------------------------------------
if not CHANGE_FILE.exists():
    raise FileNotFoundError(
        f"Missing {CHANGE_FILE}\n"
        "Run run_esol_analysis.py first."
    )

changes = pd.read_csv(CHANGE_FILE)

required = {
    "conformers", "target", "rotatable_bonds", "flexibility",
    "SIL_prediction", "MIL_prediction",
    "abs_prediction_change", "abs_error_SIL", "abs_error_MIL",
    "error_improvement"
}
missing = required - set(changes.columns)
if missing:
    raise RuntimeError(f"prediction_changes.csv is missing columns: {sorted(missing)}")

changes["rotatable_bonds"] = pd.to_numeric(changes["rotatable_bonds"], errors="coerce")
changes["error_improvement"] = pd.to_numeric(changes["error_improvement"], errors="coerce")
changes["abs_prediction_change"] = pd.to_numeric(changes["abs_prediction_change"], errors="coerce")

# ---------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------
corr_rows = []

for nconf in [5, 10, 15, 20]:
    sub = changes[changes["conformers"] == nconf].dropna(
        subset=["rotatable_bonds", "error_improvement", "abs_prediction_change"]
    )

    for variable in ["error_improvement", "abs_prediction_change"]:
        x = sub["rotatable_bonds"].to_numpy()
        y = sub[variable].to_numpy()

        if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            pr = pearsonr(x, y)
            sr = spearmanr(x, y)
            corr_rows.append({
                "Conformers": nconf,
                "Variable": variable,
                "N": len(x),
                "Pearson_r": pr.statistic,
                "Pearson_p": pr.pvalue,
                "Spearman_rho": sr.statistic,
                "Spearman_p": sr.pvalue,
            })

correlations = pd.DataFrame(corr_rows)
correlations.to_csv(OUT / "flexibility_correlations.csv", index=False)

# ---------------------------------------------------------------------
# Group summaries
# ---------------------------------------------------------------------
flex_order = ["0", "1-3", "4-6", ">=7"]
group_rows = []

for nconf in [5, 10, 15, 20]:
    suball = changes[changes["conformers"] == nconf]

    for group in flex_order:
        sub = suball[suball["flexibility"] == group].copy()
        if len(sub) == 0:
            continue

        improved = sub["error_improvement"] > 0
        worsened = sub["error_improvement"] < 0
        unchanged = sub["error_improvement"] == 0

        group_rows.append({
            "Conformers": nconf,
            "Flexibility": group,
            "N": len(sub),
            "Mean_RotB": sub["rotatable_bonds"].mean(),
            "Median_RotB": sub["rotatable_bonds"].median(),
            "Mean_error_improvement": sub["error_improvement"].mean(),
            "Median_error_improvement": sub["error_improvement"].median(),
            "Mean_abs_prediction_change": sub["abs_prediction_change"].mean(),
            "Median_abs_prediction_change": sub["abs_prediction_change"].median(),
            "Fraction_improved": improved.mean(),
            "Fraction_worsened": worsened.mean(),
            "Fraction_unchanged": unchanged.mean(),
            "Mean_abs_error_SIL": sub["abs_error_SIL"].mean(),
            "Mean_abs_error_MIL": sub["abs_error_MIL"].mean(),
        })

groups = pd.DataFrame(group_rows)
groups.to_csv(OUT / "improvement_by_flexibility.csv", index=False)

# Separate concise prediction-change table
prediction_group = groups[
    ["Conformers", "Flexibility", "N",
     "Mean_abs_prediction_change", "Median_abs_prediction_change"]
].copy()
prediction_group.to_csv(OUT / "prediction_change_by_flexibility.csv", index=False)

# ---------------------------------------------------------------------
# Paired error tests
# ---------------------------------------------------------------------
test_rows = []

for nconf in [5, 10, 15, 20]:
    sub = changes[changes["conformers"] == nconf].dropna(
        subset=["abs_error_SIL", "abs_error_MIL"]
    )

    diff = sub["abs_error_SIL"] - sub["abs_error_MIL"]

    # Wilcoxon signed-rank is paired and non-parametric.
    # scipy returns a statistic/p-value; zero differences are handled
    # with the "wilcox" convention.
    try:
        w = wilcoxon(
            sub["abs_error_SIL"],
            sub["abs_error_MIL"],
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        p = w.pvalue
        stat = w.statistic
    except Exception:
        p = np.nan
        stat = np.nan

    test_rows.append({
        "Conformers": nconf,
        "N": len(sub),
        "Mean_abs_error_difference_SIL_minus_MIL": diff.mean(),
        "Median_abs_error_difference": diff.median(),
        "Wilcoxon_statistic": stat,
        "Wilcoxon_p": p,
        "Fraction_MIL_improved": (diff > 0).mean(),
        "Fraction_MIL_worsened": (diff < 0).mean(),
        "Fraction_equal": (diff == 0).mean(),
    })

tests = pd.DataFrame(test_rows)
tests.to_csv(OUT / "paired_error_tests.csv", index=False)

# ---------------------------------------------------------------------
# Molecule-level summary across conformer counts
# ---------------------------------------------------------------------
wide = None

for nconf in [5, 10, 15, 20]:
    sub = changes[changes["conformers"] == nconf][
        ["row", "rotatable_bonds", "flexibility",
         "target", "SIL_prediction", "MIL_prediction",
         "error_improvement", "abs_prediction_change"]
    ].copy()

    sub = sub.rename(columns={
        "error_improvement": f"error_improvement_{nconf}conf",
        "abs_prediction_change": f"abs_prediction_change_{nconf}conf",
        "MIL_prediction": f"MIL_prediction_{nconf}conf",
    })

    if wide is None:
        wide = sub
    else:
        wide = wide.merge(
            sub.drop(columns=["rotatable_bonds", "flexibility", "target",
                              "SIL_prediction"]),
            on="row",
            how="outer"
        )

wide["mean_error_improvement_5to20"] = wide[
    [f"error_improvement_{n}conf" for n in [5, 10, 15, 20]]
].mean(axis=1)

wide["mean_abs_prediction_change_5to20"] = wide[
    [f"abs_prediction_change_{n}conf" for n in [5, 10, 15, 20]]
].mean(axis=1)

wide.to_csv(OUT / "molecule_level_summary.csv", index=False)

# ---------------------------------------------------------------------
# Summary across all flexible molecules
# ---------------------------------------------------------------------
summary_rows = []

for nconf in [5, 10, 15, 20]:
    sub = changes[changes["conformers"] == nconf]

    # Flexible = >=4 rotatable bonds
    flex = sub[sub["rotatable_bonds"] >= 4]

    summary_rows.append({
        "Conformers": nconf,
        "N_total": len(sub),
        "N_RotB_ge4": len(flex),
        "Fraction_RotB_ge4": len(flex) / len(sub),
        "Mean_error_improvement_all": sub["error_improvement"].mean(),
        "Median_error_improvement_all": sub["error_improvement"].median(),
        "Fraction_improved_all": (sub["error_improvement"] > 0).mean(),
        "Mean_error_improvement_RotB_ge4": flex["error_improvement"].mean(),
        "Median_error_improvement_RotB_ge4": flex["error_improvement"].median(),
        "Fraction_improved_RotB_ge4": (flex["error_improvement"] > 0).mean(),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "flexibility_summary.csv", index=False)

# ---------------------------------------------------------------------
# FIGURE 1: error improvement vs RotB
# ---------------------------------------------------------------------
for nconf in [5, 10, 15, 20]:
    sub = changes[changes["conformers"] == nconf]

    plt.figure(figsize=(7, 5))
    plt.scatter(
        sub["rotatable_bonds"],
        sub["error_improvement"],
        alpha=0.65
    )
    plt.axhline(0, linestyle="--")
    plt.xlabel("Number of rotatable bonds")
    plt.ylabel("Error improvement (SIL absolute error − MIL absolute error)")
    plt.title(f"ESOL: molecule-level error improvement, {nconf} conformers")
    plt.tight_layout()
    plt.savefig(FIG / f"error_improvement_vs_rotb_{nconf}conf.png", dpi=300)
    plt.close()

# ---------------------------------------------------------------------
# FIGURE 2: absolute prediction change vs RotB
# ---------------------------------------------------------------------
for nconf in [5, 10, 15, 20]:
    sub = changes[changes["conformers"] == nconf]

    plt.figure(figsize=(7, 5))
    plt.scatter(
        sub["rotatable_bonds"],
        sub["abs_prediction_change"],
        alpha=0.65
    )
    plt.xlabel("Number of rotatable bonds")
    plt.ylabel("|MIL prediction − SIL prediction|")
    plt.title(f"ESOL: prediction shift relative to SIL, {nconf} conformers")
    plt.tight_layout()
    plt.savefig(FIG / f"prediction_change_vs_rotb_{nconf}conf.png", dpi=300)
    plt.close()

# ---------------------------------------------------------------------
# FIGURE 3: fraction improved by flexibility group
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
for nconf in [5, 10, 15, 20]:
    sub = groups[groups["Conformers"] == nconf].copy()
    sub["Flexibility"] = pd.Categorical(
        sub["Flexibility"], flex_order, ordered=True
    )
    sub = sub.sort_values("Flexibility")
    plt.plot(
        sub["Flexibility"].astype(str),
        sub["Fraction_improved"],
        marker="o",
        label=f"{nconf} conf"
    )

plt.ylim(0, 1)
plt.xlabel("Rotatable-bond group")
plt.ylabel("Fraction of molecules improved")
plt.title("ESOL: fraction improved relative to SIL")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "improvement_fraction_vs_flexibility.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# FIGURE 4: mean error improvement by flexibility
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
for nconf in [5, 10, 15, 20]:
    sub = groups[groups["Conformers"] == nconf].copy()
    sub["Flexibility"] = pd.Categorical(
        sub["Flexibility"], flex_order, ordered=True
    )
    sub = sub.sort_values("Flexibility")
    plt.plot(
        sub["Flexibility"].astype(str),
        sub["Mean_error_improvement"],
        marker="o",
        label=f"{nconf} conf"
    )

plt.axhline(0, linestyle="--")
plt.xlabel("Rotatable-bond group")
plt.ylabel("Mean error improvement")
plt.title("ESOL: mean molecule-level error improvement")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "rmse_improvement_vs_flexibility.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
lines = []
lines.append("=" * 80)
lines.append("ESOL MOLECULE-LEVEL FLEXIBILITY ANALYSIS")
lines.append("=" * 80)
lines.append("")
lines.append("Purpose:")
lines.append(
    "Test whether the benefit/change introduced by conformer-based QSARmil "
    "relative to conventional SIL is associated with molecular flexibility."
)
lines.append("")
lines.append("Design:")
lines.append(
    "One fixed ESOL train/test split (test_size=0.20, random_state=42); "
    "226 test molecules."
)
lines.append("")
lines.append("CAUTION:")
lines.append(
    "This is exploratory evidence from one split. Correlation does not "
    "establish causality, and subgroup sizes—especially >=7 rotatable bonds—"
    "are small."
)
lines.append("")

lines.append("FLEXIBILITY SUMMARY")
lines.append("-" * 80)
lines.append(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
lines.append("")

lines.append("CORRELATIONS WITH ROTATABLE BONDS")
lines.append("-" * 80)
lines.append(correlations.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
lines.append("")

lines.append("FLEXIBILITY-GROUP RESULTS")
lines.append("-" * 80)
lines.append(groups.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
lines.append("")

lines.append("PAIRED ERROR TESTS")
lines.append("-" * 80)
lines.append(tests.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
lines.append("")

lines.append("INTERPRETATION GUIDE")
lines.append("-" * 80)
lines.append(
    "error_improvement > 0 means the MIL prediction has smaller absolute "
    "error than SIL for that molecule."
)
lines.append(
    "error_improvement < 0 means SIL has smaller absolute error."
)
lines.append(
    "A positive correlation between RotB and error_improvement would be "
    "consistent with greater molecule-level benefit from conformer-based "
    "prediction as flexibility increases; a negative correlation would "
    "indicate the opposite. Statistical significance must be interpreted "
    "with multiple testing and the single-split limitation in mind."
)
lines.append("")
lines.append("Generated files:")
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        lines.append(str(p))

(OUT / "ESOL_FLEXIBILITY_REPORT.txt").write_text(
    "\n".join(lines), encoding="utf-8"
)

print("=" * 80)
print("ESOL FLEXIBILITY ANALYSIS COMPLETE")
print("=" * 80)
print(f"Output directory: {OUT}")
print("")
print("FLEXIBILITY SUMMARY")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("")
print("CORRELATIONS")
print(correlations.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("")
print("PAIRED ERROR TESTS")
print(tests.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("")
print(f"Report: {OUT/'ESOL_FLEXIBILITY_REPORT.txt'}")
