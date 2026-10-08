import os
import time
import pandas as pd
import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from qsarmil.modelling.meta import MultiConformerRegressor


# ============================================================
# FreeSolv — QSARmil conformer-number scaling
# ============================================================

TRAIN = "data/freesolv_train_seed42.csv"
TEST = "data/freesolv_test_seed42.csv"

CONFORMER_COUNTS = [5]

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

smiles_train = train["smiles"].tolist()
smiles_test = test["smiles"].tolist()

y_train = train[
    "experimental_dG_hyd_kcal_mol"
].values

y_test = test[
    "experimental_dG_hyd_kcal_mol"
].values


results = []


for nconf in CONFORMER_COUNTS:

    print("\n")
    print("=" * 75)
    print(f"FreeSolv — QSARmil — {nconf} conformers")
    print("=" * 75)

    output_folder = f"freesolv_qsarmil_{nconf}conf"

    start = time.time()

    model = MultiConformerRegressor(
        num_conf=nconf,
        hopt=False,
        verbose=True,
        random_seed=42,
        accelerator="cpu",
        output_folder=output_folder
    )

    prediction = model.train_predict(
        smiles_train,
        y_train,
        smiles_test
    )

    runtime = time.time() - start

    prediction = np.asarray(prediction)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    r2 = r2_score(y_test, prediction)

    rmse = np.sqrt(
        mean_squared_error(y_test, prediction)
    )

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    corr = np.corrcoef(
        y_test,
        prediction
    )[0, 1]

    print("\n")
    print("-" * 75)
    print(f"RESULT — {nconf} conformers")
    print("-" * 75)

    print(f"N       = {len(y_test)}")
    print(f"R²      = {r2:.4f}")
    print(f"RMSE    = {rmse:.4f}")
    print(f"MAE     = {mae:.4f}")
    print(f"r       = {corr:.4f}")
    print(f"Runtime = {runtime / 60:.2f} min")

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    pred_df = test.copy()

    pred_df["prediction"] = prediction
    pred_df["num_conformers"] = nconf

    pred_file = f"freesolv_qsarmil_{nconf}conf_predictions.csv"

    pred_df.to_csv(
        pred_file,
        index=False
    )

    print(f"\nSaved: {pred_file}")

    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    results.append({
        "conformers": nconf,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson_r": corr,
        "runtime_min": runtime / 60
    })

    # Save progressively so a long run is not lost
    pd.DataFrame(results).to_csv(
        "freesolv_conformer_scaling_summary.csv",
        index=False
    )


# ============================================================
# Final summary
# ============================================================

summary = pd.DataFrame(results)

print("\n\n")
print("=" * 75)
print("FreeSolv — CONFORMER SCALING SUMMARY")
print("=" * 75)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved:")
print("freesolv_conformer_scaling_summary.csv")
