import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

BASE = os.path.expanduser("~/Documents/qsarmil")
INFILE = os.path.join(
    BASE,
    "freesolv_descriptor_domain_analysis",
    "freesolv_test_domain_vs_model_errors.csv"
)

OUTDIR = os.path.join(
    BASE,
    "freesolv_descriptor_domain_analysis"
)

df = pd.read_csv(INFILE)

models = {
    "MORSE": "MORSE_MeanInstanceWrapper_improvement",
    "Pmapper": "Pmapper_MeanInstanceWrapper_improvement",
    "ElectroShape": "ElectroShape_MeanInstanceWrapper_improvement",
    "GETAWAY": "GETAWAY_MeanInstanceWrapper_improvement",
}

metrics = [
    "ensemble_spread",
    "nearest_train_distance",
    "mean_train_distance",
]

all_results = []
pair_results = []

for descriptor in models:

    model_col = models[descriptor]

    sub = df[
        df["descriptor"] == descriptor
    ].copy()

    print("\n======================================")
    print(descriptor)
    print("======================================")

    for metric in metrics:

        sub2 = sub[
            [metric, model_col]
        ].dropna().copy()

        if len(sub2) < 20:
            continue

        # Four approximately equal-sized groups
        sub2["quartile"] = pd.qcut(
            sub2[metric],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop"
        )

        print("\nMetric:", metric)

        groups = []

        for q in ["Q1", "Q2", "Q3", "Q4"]:

            values = sub2.loc[
                sub2["quartile"] == q,
                model_col
            ].values

            if len(values) == 0:
                continue

            groups.append(values)

            row = {
                "descriptor": descriptor,
                "metric": metric,
                "quartile": q,
                "n": len(values),
                "mean_improvement": np.mean(values),
                "median_improvement": np.median(values),
                "std_improvement": np.std(values, ddof=1),
                "fraction_improved": np.mean(values > 0),
                "q25": np.percentile(values, 25),
                "q75": np.percentile(values, 75),
            }

            all_results.append(row)

            print(
                f"{q}: n={len(values):3d} "
                f"mean={np.mean(values): .4f} "
                f"median={np.median(values): .4f} "
                f"improved={np.mean(values > 0):.3f}"
            )

        # Kruskal-Wallis test across quartiles
        if len(groups) >= 3:
            H, p = kruskal(*groups)

            print(
                f"Kruskal-Wallis: H={H:.4f}, p={p:.6g}"
            )

            pair_results.append({
                "descriptor": descriptor,
                "metric": metric,
                "test": "Kruskal-Wallis",
                "H": H,
                "p_value": p,
                "n": len(sub2),
            })

        # Spearman correlation of quartile rank vs improvement
        rank = (
            sub2["quartile"]
            .map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})
        )

        rho, p = spearmanr(
            rank,
            sub2[model_col]
        )

        pair_results.append({
            "descriptor": descriptor,
            "metric": metric,
            "test": "Quartile-rank Spearman",
            "H": np.nan,
            "p_value": p,
            "rho": rho,
            "n": len(sub2),
        })

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

quartile_df = pd.DataFrame(all_results)

quartile_file = os.path.join(
    OUTDIR,
    "freesolv_domain_quartile_results.csv"
)

quartile_df.to_csv(
    quartile_file,
    index=False
)

tests_df = pd.DataFrame(pair_results)

tests_file = os.path.join(
    OUTDIR,
    "freesolv_domain_quartile_tests.csv"
)

tests_df.to_csv(
    tests_file,
    index=False
)

print("\n======================================")
print("SUMMARY OF QUARTILE TESTS")
print("======================================")

print(
    tests_df.to_string(index=False)
)

print("\nSaved:")
print(quartile_file)
print(tests_file)

# ------------------------------------------------------------
# Identify largest Q1 -> Q4 changes
# ------------------------------------------------------------

print("\n======================================")
print("Q1 -> Q4 CHANGES")
print("======================================")

for descriptor in models:

    for metric in metrics:

        x = quartile_df[
            (quartile_df["descriptor"] == descriptor) &
            (quartile_df["metric"] == metric)
        ]

        if len(x) < 4:
            continue

        q1 = x.loc[
            x["quartile"] == "Q1",
            "mean_improvement"
        ]

        q4 = x.loc[
            x["quartile"] == "Q4",
            "mean_improvement"
        ]

        if len(q1) and len(q4):

            delta = q4.iloc[0] - q1.iloc[0]

            print(
                f"{descriptor:14s} "
                f"{metric:24s} "
                f"Q4-Q1 = {delta: .4f}"
            )
