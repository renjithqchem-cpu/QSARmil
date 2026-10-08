import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

f = "esol_qsarmil_3conf_predictions.csv"
df = pd.read_csv(f)

y = df["measured log solubility in mols per litre"]
p = df["prediction"]

r2 = r2_score(y, p)
rmse = np.sqrt(mean_squared_error(y, p))
mae = mean_absolute_error(y, p)

print("=" * 60)
print("ESOL — QSARmil 3-conformer external test")
print("=" * 60)
print(f"N    = {len(df)}")
print(f"R²   = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")

# Flexibility-stratified performance
print("\nFlexibility-stratified performance")
print("-" * 60)

bins = [-1, 0, 3, 6, 100]
labels = ["0 RotB", "1–3 RotB", "4–6 RotB", "≥7 RotB"]

df["Flexibility"] = pd.cut(
    df["RotB"],
    bins=bins,
    labels=labels
)

for group, sub in df.groupby("Flexibility", observed=False):
    if len(sub) == 0:
        continue

    yy = sub["measured log solubility in mols per litre"]
    pp = sub["prediction"]

    print(
        f"{str(group):10s} "
        f"N={len(sub):3d}  "
        f"R²={r2_score(yy, pp):7.4f}  "
        f"RMSE={np.sqrt(mean_squared_error(yy, pp)):7.4f}  "
        f"MAE={mean_absolute_error(yy, pp):7.4f}"
    )

print("\nPrediction–experimental correlation:")
print(np.corrcoef(y, p)[0, 1])

# Save analyzed predictions
df.to_csv("esol_qsarmil_3conf_analyzed.csv", index=False)
