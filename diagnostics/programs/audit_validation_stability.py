import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

val = pd.read_csv("freesolv_qsarmil_5conf/val.csv")
train = pd.read_csv("freesolv_qsarmil_5conf/train.csv")

models = [c for c in val.columns if c not in ["SMILES", "Y_TRUE"]]

print("=" * 100)
print("TEST-BLIND VALIDATION STABILITY FORENSICS")
print("=" * 100)
print("Training rows:", len(train))
print("Validation rows:", len(val))
print("Models:", len(models))

rows = []

for m in models:

    y = val["Y_TRUE"].to_numpy()
    p = val[m].to_numpy()
    residual = y - p

    train_pred_available = m in train.columns

    if train_pred_available:
        tp = train[m].to_numpy()

        train_min = np.nanmin(tp)
        train_max = np.nanmax(tp)

        val_min = np.nanmin(p)
        val_max = np.nanmax(p)

        outside = np.sum((p < train_min) | (p > train_max))
        outside_fraction = outside / len(p)

        train_mean = np.nanmean(tp)
        train_std = np.nanstd(tp)

        val_mean = np.nanmean(p)
        val_std = np.nanstd(p)

        mean_shift = val_mean - train_mean
        std_ratio = val_std / train_std if train_std > 0 else np.nan

    else:
        train_min = np.nan
        train_max = np.nan
        val_min = np.nanmin(p)
        val_max = np.nanmax(p)
        outside = np.nan
        outside_fraction = np.nan
        train_mean = np.nan
        train_std = np.nan
        val_mean = np.nanmean(p)
        val_std = np.nanstd(p)
        mean_shift = np.nan
        std_ratio = np.nan

    rows.append({
        "model": m,
        "val_RMSE": np.sqrt(mean_squared_error(y, p)),
        "val_MAE": mean_absolute_error(y, p),
        "val_prediction_min": val_min,
        "val_prediction_max": val_max,
        "val_prediction_mean": val_mean,
        "val_prediction_std": val_std,
        "train_prediction_min": train_min,
        "train_prediction_max": train_max,
        "train_prediction_mean": train_mean,
        "train_prediction_std": train_std,
        "val_values_outside_train_prediction_range": outside,
        "val_outside_fraction": outside_fraction,
        "validation_mean_shift": mean_shift,
        "validation_std_ratio": std_ratio,
        "max_abs_residual": np.max(np.abs(residual)),
        "residual_std": np.std(residual)
    })

a = pd.DataFrame(rows)

print()
print("=" * 100)
print("1. MODELS WITH LARGEST VALIDATION PREDICTION-RANGE EXTRAPOLATION")
print("=" * 100)

print(
    a.sort_values("val_outside_fraction", ascending=False).head(15).to_string(index=False)
)

print()
print("=" * 100)
print("2. MODELS WITH LARGEST ABSOLUTE VALIDATION RESIDUALS")
print("=" * 100)

print(
    a.sort_values("max_abs_residual", ascending=False).head(15).to_string(index=False)
)

print()
print("=" * 100)
print("3. MODELS WITH LARGEST TRAIN -> VALIDATION MEAN SHIFT")
print("=" * 100)

a["abs_mean_shift"] = a["validation_mean_shift"].abs()

print(
    a.sort_values("abs_mean_shift", ascending=False).head(15).to_string(index=False)
)

print()
print("=" * 100)
print("4. MODELS WITH LARGEST TRAIN -> VALIDATION STD CHANGE")
print("=" * 100)

a["abs_log_std_ratio"] = np.abs(np.log(a["validation_std_ratio"]))

print(
    a.sort_values("abs_log_std_ratio", ascending=False).head(15).to_string(index=False)
)

print()
print("=" * 100)
print("5. QSARCONS SELECTED MODELS")
print("=" * 100)

selected = [
    "MolFeatElectroShape|MeanInstanceWrapperMLPNetworkRegressor",
    "RDKitMORSE|MeanInstanceWrapperMLPNetworkRegressor",
    "MolFeatPmapper|MeanInstanceWrapperMLPNetworkRegressor",
    "RDKitGETAWAY|MeanInstanceWrapperMLPNetworkRegressor"
]

print(
    a[a["model"].isin(selected)].to_string(index=False)
)

print()
print("=" * 100)
print("6. GETAWAY VS OTHER SELECTED MODELS")
print("=" * 100)

cols = [
    "model",
    "val_RMSE",
    "val_prediction_min",
    "val_prediction_max",
    "train_prediction_min",
    "train_prediction_max",
    "val_values_outside_train_prediction_range",
    "val_outside_fraction",
    "validation_mean_shift",
    "validation_std_ratio",
    "max_abs_residual",
    "residual
]

print(
    a[a["
)

print()
print("
print("7. CORRELATIONS BETWEEN VALIDATION RMSE AND STABILITY INDICA


numeric_checks = [
    "val_outside_fraction",
   
    "abs_log_std_ratio",
    "max_abs_residual",
    "r


for c in numeric_checks:
    x = a["val_RMSE"]
    z = a[c]

    

    if mask.sum(
        r = np.corrcoef(x[mask], z[mask])[0, 1]
  

print()
print("
print("8. MOST EXTREME VALIDATION P
print("=" * 100)

for m in selected

    p = val[m]

    imin = p


    prin

    print("Minim

    print("Y_TRUE:", val.loc[imi

    print("Maximum predi
    print("SMILES:",


outfile = "freesolv_qsarmil_5conf/validation_stability_forensics.csv"
a.to_csv(outfile, index=False)

print()
print("=" * 100)
print("AUDIT COMPLETE")
print("=" * 100)
print("Saved:", outfile)
