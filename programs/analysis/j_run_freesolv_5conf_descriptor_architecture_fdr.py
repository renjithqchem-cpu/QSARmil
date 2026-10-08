from pathlib import Path
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path.home() / "Documents" / "qsarmil"
IN_DIR = ROOT / "data" / "freesolv_descriptor_architecture_interaction"
OUT_DIR = IN_DIR / "fdr_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

desc_file = IN_DIR / "paired_descriptor_comparisons.csv"
arch_file = IN_DIR / "paired_architecture_comparisons.csv"

desc = pd.read_csv(desc_file)
arch = pd.read_csv(arch_file)

def apply_bh(df):
    df = df.copy()
    p = df["wilcoxon_p"].to_numpy(dtype=float)
    reject, q, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    df["BH_q"] = q
    df["significant_fdr_0.05"] = reject
    return df.sort_values(["BH_q", "wilcoxon_p"]).reset_index(drop=True)

desc = apply_bh(desc)
arch = apply_bh(arch)

desc.to_csv(OUT_DIR / "paired_descriptor_comparisons_FDR.csv", index=False)
arch.to_csv(OUT_DIR / "paired_architecture_comparisons_FDR.csv", index=False)

summary = []
for label, df in [("descriptor", desc), ("architecture", arch)]:
    summary.append({
        "comparison_type": label,
        "total_tests": len(df),
        "raw_p_lt_0.05": int((df.wilcoxon_p < 0.05).sum()),
        "FDR_q_lt_0.05": int(df["significant_fdr_0.05"].sum()),
        "minimum_raw_p": float(df.wilcoxon_p.min()),
        "minimum_BH_q": float(df.BH_q.min())
    })
overall = pd.DataFrame(summary)
overall.to_csv(OUT_DIR / "FDR_overall_summary.csv", index=False)

by_count = []
if "n_conformers" in desc.columns:
    for n, g in desc.groupby("n_conformers", sort=True):
        by_count.append({
            "comparison_type": "descriptor",
            "n_conformers": n,
            "total_tests": len(g),
            "raw_p_lt_0.05": int((g.wilcoxon_p < 0.05).sum()),
            "FDR_q_lt_0.05": int(g["significant_fdr_0.05"].sum()),
            "minimum_raw_p": float(g.wilcoxon_p.min()),
            "minimum_BH_q": float(g.BH_q.min())
        })
if "n_conformers" in arch.columns:
    for n, g in arch.groupby("n_conformers", sort=True):
        by_count.append({
            "comparison_type": "architecture",
            "n_conformers": n,
            "total_tests": len(g),
            "raw_p_lt_0.05": int((g.wilcoxon_p < 0.05).sum()),
            "FDR_q_lt_0.05": int(g["significant_fdr_0.05"].sum()),
            "minimum_raw_p": float(g.wilcoxon_p.min()),
            "minimum_BH_q": float(g.BH_q.min())
        })
by_count_df = pd.DataFrame(by_count)
if len(by_count_df):
    by_count_df.to_csv(OUT_DIR / "FDR_summary_by_conformer.csv", index=False)

desc.head(30).to_csv(OUT_DIR / "top30_descriptor_effects_FDR.csv", index=False)
arch.head(30).to_csv(OUT_DIR / "top30_architecture_effects_FDR.csv", index=False)

print("\n" + "="*80)
print("FREESOLV 5-CONFORMER BENJAMINI-HOCHBERG FDR ANALYSIS")
print("="*80)

print("\nOVERALL SUMMARY")
print("-"*80)
print(overall.to_string(index=False))

if len(by_count_df):
    print("\nSUMMARY BY CONFORMER COUNT")
    print("-"*80)
    print(by_count_df.to_string(index=False))

print("\nTOP 30 DESCRIPTOR COMPARISONS AFTER FDR")
print("-"*80)
print(desc.head(30).to_string(index=False))

print("\nTOP 30 ARCHITECTURE COMPARISONS AFTER FDR")
print("-"*80)
print(arch.head(30).to_string(index=False))

print("\nOUTPUT")
print(OUT_DIR)
print("\nFDR analysis completed successfully.")

