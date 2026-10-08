import random
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from qsarcons.consensus import GeneticSearch, SystematicSearch

VAL = "freesolv_qsarmil_5conf/val.csv"

df = pd.read_csv(VAL)

y = df["Y_TRUE"].astype(float)
X = df[[c for c in df.columns if "|" in c]].copy()

print("=" * 100)
print("QSARCONS GENETIC SEARCH FORENSIC AUDIT")
print("=" * 100)

print(f"\nValidation molecules: {len(y)}")
print(f"Candidate models:     {X.shape[1]}")

# ------------------------------------------------------------------
# Individual model R2 filtering
# ------------------------------------------------------------------

r2 = {}

for c in X.columns:
    r2[c] = r2_score(y, X[c])

r2s = pd.Series(r2).sort_values(ascending=False)

print("\nModels with validation R2 > 0:",
      int((r2s > 0).sum()))
print("Models with validation R2 <= 0:",
      int((r2s <= 0).sum()))

print("\nTop 15 individual validation models:")
print(r2s.head(15).to_string())

# ------------------------------------------------------------------
# Actual QSARcons GeneticSearch
# ------------------------------------------------------------------

print("\n" + "-" * 100)
print("ACTUAL QSARCONS GENETIC SEARCH")
print("-" * 100)

for size in [2,4,6,8,10,12,14]:

    gs = GeneticSearch(cons_size=size, n_iter=50)

    selected = gs.run(X, y)

    pred = X[selected].mean(axis=1)

    score = r2_score(y, pred)
    rmse = root_mean_squared_error(y, pred)

    print(
        f"\nSize {size:2d}: "
        f"R2={score:.8f}  "
        f"RMSE={rmse:.8f}"
    )

    print("Selected:")
    for m in selected:
        print("  ", m)

# ------------------------------------------------------------------
# Automatic QSARcons search
# ------------------------------------------------------------------

print("\n" + "-" * 100)
print("QSARCONS AUTO SEARCH")
print("-" * 100)

ga_auto = GeneticSearch(cons_size="auto", n_iter=50)

auto_selected = ga_auto.run(X, y)

auto_pred = X[auto_selected].mean(axis=1)

print("\nAUTO selected models:")
for m in auto_selected:
    print("  ", m)

print(
    "\nAUTO validation R2 =",
    r2_score(y, auto_pred)
)

print(
    "AUTO validation RMSE =",
    root_mean_squared_error(y, auto_pred)
)

# ------------------------------------------------------------------
# Systematic top-k
# ------------------------------------------------------------------

print("\n" + "-" * 100)
print("SYSTEMATIC TOP-K BY INDIVIDUAL VALIDATION R2")
print("-" * 100)

for k in [2,4,6,8,10,12,14]:

    cols = r2s.head(k).index.tolist()

    pred = X[cols].mean(axis=1)

    print(
        f"k={k:2d}: "
        f"R2={r2_score(y,pred):.8f}  "
        f"RMSE={root_mean_squared_error(y,pred):.8f}"
    )

# ------------------------------------------------------------------
# Random subset search
# ------------------------------------------------------------------

print("\n" + "-" * 100)
print("RANDOM SUBSET SEARCH")
print("-" * 100)

rng = random.Random(42)

candidate_cols = r2s[r2s > 0].index.tolist()

print("Candidate pool after R2>0 filtering:", len(candidate_cols))

for k in [2,4,6,8,10,12,14]:

    best_r2 = -np.inf
    best_cols = None

    n_iter = 10000

    for _ in range(n_iter):

        cols = rng.sample(candidate_cols, k)

        pred = X[cols].mean(axis=1)

        score = r2_score(y, pred)

        if score > best_r2:
            best_r2 = score
            best_cols = cols

    best_rmse = root_mean_squared_error(
        y,
        X[best_cols].mean(axis=1)
    )

    print(
        f"\nk={k:2d}: "
        f"best_random_R2={best_r2:.8f}  "
        f"best_random_RMSE={best_rmse:.8f}"
    )

    print("Best random subset:")
    for m in best_cols:
        print("  ", m)

print("\nAudit complete.")
