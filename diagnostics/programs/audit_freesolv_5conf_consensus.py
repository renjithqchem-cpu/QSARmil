import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

VAL = "freesolv_qsarmil_5conf/val.csv"
TEST = "freesolv_qsarmil_5conf/test.csv"
TARGET = "data/freesolv_test_seed42.csv"

val = pd.read_csv(VAL)
test = pd.read_csv(TEST)
target = pd.read_csv(TARGET)

y_val = val["Y_TRUE"].to_numpy(float)

target_map = dict(
    zip(
        target["smiles"].astype(str),
        target["experimental_dG_hyd_kcal_mol"].astype(float)
    )
)

y_test = test["SMILES"].astype(str).map(target_map).to_numpy(float)

model_cols = [c for c in val.columns if "|" in c]

rows = []

for col in model_cols:
    pv = val[col].to_numpy(float)
    pt = test[col].to_numpy(float)

    rows.append({
        "model": col,
        "val_RMSE": mean_squared_error(y_val, pv) ** 0.5,
        "val_MAE": mean_absolute_error(y_val, pv),
        "test_RMSE": mean_squared_error(y_test, pt) ** 0.5,
        "test_MAE": mean_absolute_error(y_test, pt),
    })

df = pd.DataFrame(rows)

print("\n" + "="*100)
print("FREESOLV 5-CONFORMER CONSENSUS FORENSICS")
print("="*100)

rho, p = spearmanr(df["val_RMSE"], df["test_RMSE"])

print(f"\nValidation-vs-test RMSE Spearman rho = {rho:.4f}")
print(f"p-value = {p:.4g}")

print("\n" + "-"*100)
print("QSARmil GENETIC CONSENSUS MEMBERS")
print("-"*100)

selected = [
    "MolFeatElectroShape|MeanInstanceWrapperMLPNetworkRegressor",
    "RDKitMORSE|MeanInstanceWrapperMLPNetworkRegressor",
    "MolFeatPmapper|MeanInstanceWrapperMLPNetworkRegressor",
    "RDKitGETAWAY|MeanInstanceWrapperMLPNetworkRegressor",
]

print(
    df[df["model"].isin(selected)]
    .sort_values("test_RMSE")
    .to_string(index=False)
)

print("\n" + "-"*100)
print("TOP 15 BY VALIDATION RMSE")
print("-"*100)
print(
    df.sort_values("val_RMSE")
      .head(15)
      .to_string(index=False)
)

print("\n" + "-"*100)
print("TOP 15 BY TEST RMSE")
print("-"*100)
print(
    df.sort_values("test_RMSE")
      .head(15)
      .to_string(index=False)
)

print("\n" + "-"*100)
print("VALIDATION-RANKED SIMPLE ENSEMBLES")
print("-"*100)

for k in [2, 3, 4, 5, 10]:

    selected_k = (
        df.sort_values("val_RMSE")
          .head(k)["model"]
          .tolist()
    )

    pred = test[selected_k].mean(axis=1).to_numpy(float)

    print(
        f"Top-{k}: "
        f"RMSE={mean_squared_error(y_test,pred)**0.5:.4f}  "
        f"MAE={mean_absolute_error(y_test,pred):.4f}"
    )

print("\n" + "-"*100)
print("DESCRIPTIVE ORACLE TEST-RANKED ENSEMBLES")
print("(NOT usable for model selection; shown only to diagnose ensemble capacity)")
print("-"*100)

for k in [2, 3, 4, 5, 10]:

    selected_k = (
        df.sort_values("test_RMSE")
          .head(k)["model"]
          .tolist()
    )

    pred = test[selected_k].mean(axis=1).to_numpy(float)

    print(
        f"Test-top-{k}: "
        f"RMSE={mean_squared_error(y_test,pred)**0.5:.4f}  "
        f"MAE={mean_absolute_error(y_test,pred):.4f}"
    )

print("\n" + "-"*100)
print("Pmapper-only ensemble")
print("-"*100)

pmapper = [
    c for c in model_cols
    if c.startswith("MolFeatPmapper|")
]

pred = test[pmapper].mean(axis=1).to_numpy(float)

print(f"Models = {len(pmapper)}")
print(f"RMSE = {mean_squared_error(y_test,pred)**0.5:.4f}")
print(f"MAE  = {mean_absolute_error(y_test,pred):.4f}")

print("\n" + "-"*100)
print("ALL MeanInstanceWrapperMLP models")
print("-"*100)

wrapper = [
    c for c in model_cols
    if "MeanInstanceWrapperMLP" in c
]

pred = test[wrapper].mean(axis=1).to_numpy(float)

print(f"Models = {len(wrapper)}")
print(f"RMSE = {mean_squared_error(y_test,pred)**0.5:.4f}")
print(f"MAE  = {mean_absolute_error(y_test,pred):.4f}")

# Save complete ranking
df.sort_values("test_RMSE").to_csv(
    "data/freesolv_5conf_consensus_forensics.csv",
    index=False
)

print("\nSaved:")
print("data/freesolv_5conf_consensus_forensics.csv")
