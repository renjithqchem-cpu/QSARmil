import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

f = "data/esol_3d_sil_hopt_vs_3d_mil/paired_model_comparison.csv"
d = pd.read_csv(f)

d["delta"] = d["SIL_RMSE"] - d["MIL_RMSE"]

print("=" * 70)
print("HPO 3D-SIL vs 3D-MIL STATISTICS")
print("=" * 70)
print("Rows:", len(d))
print("Models per conformer count:", d.groupby("n_conformers").size().to_dict())

out = []

for n in sorted(d.n_conformers.unique()):
    x = d.loc[d.n_conformers == n, "delta"].dropna().values

    w = wilcoxon(x)
    p = binomtest((x > 0).sum(), (x != 0).sum()).pvalue

    out.append([
        n,
        len(x),
        x.mean(),
        np.median(x),
        x.std(ddof=1),
        np.percentile(x, 2.5),
        np.percentile(x, 97.5),
        (x > 0).sum(),
        (x < 0).sum(),
        w.pvalue,
        p
    ])

r = pd.DataFrame(
    out,
    columns=[
        "n_conf",
        "n_models",
        "mean_delta_RMSE",
        "median_delta_RMSE",
        "SD_delta_RMSE",
        "P2.5",
        "P97.5",
        "improved",
        "worsened",
        "wilcoxon_p",
        "sign_test_p"
    ]
)

print()
print(r.to_string(index=False))

o = "data/esol_3d_sil_hopt_vs_3d_mil/statistics"
import os
os.makedirs(o, exist_ok=True)

r.to_csv(
    o + "/primary_statistics.csv",
    index=False
)

print()
print("Saved:", o + "/primary_statistics.csv")
print("DONE")
