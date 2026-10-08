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
# ESOL — QSARmil conformer-number scaling
# Restart-safe version
# ============================================================

TRAIN = "esol_train.csv"
TEST = "esol_test.csv"

CONFORMER_COUNTS = [5, 10, 15, 20, 25]

SUMMARY_FILE = "esol_conformer_scaling_summary.csv"


# ============================================================
# Load data
# ============================================================

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

smiles_train = train["smiles"].tolist()
smiles_test = test["smiles"].tolist()

y_train = train[
    "measured log solubility in mols per litre"
].values

y_test = test[
    "measured log solubility in mols per litre"
].values


# ============================================================
# Load existing results if available
# ============================================================

if os.path.exists(SUMMARY_FILE):

    summary = pd.read_csv(SUMMARY_FILE)

    print("\nExisting scaling results found:")
    print(summary.to_string(index=False))

else:

    summary = pd.DataFrame(
        columns=[
            "conformers",
            "R2",
            "RMSE",
            "MAE",
            "Pearson_r",
            "runtime_min"
        ]
    )


# ============================================================
# Run conformer scaling
# ============================================================

for nconf in CONFORMER_COUNTS:

    pred_file = f"esol_qsarmil_{nconf}conf_predictions.csv"

    # --------------------------------------------------------
    # Check whether this conformer count is already complete
    # --------------------------------------------------------

    already_done = (
        nconf in summary["conformers"].values
        and os.path.exists(pred_file)
    )

    if already_done:

        print("\n")
        print("=" * 75)
        print(f"SKIPPING {nconf} CONFORMERS — ALREADY COMPLETED")
        print("=" * 75)

        continue


    # --------------------------------------------------------
    # Start calculation
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print(f"ESOL — QSARmil — {nconf} conformers")
    print("=" * 75)

    output_folder = f"esol_qsarmil_{nconf}conf"

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


    # ========================================================
    # Metrics
    # ========================================================

    r2 = r2_score(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    corr = np.corrcoef(
        y_test,
        prediction
    )[0, 1]


    # ========================================================
    # Print result
    # ========================================================

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


    # ========================================================
    # Save predictions
    # ========================================================

    pred_df = test.copy()

    pred_df["prediction"] = prediction
    pred_df["num_conformers"] = nconf

    pred_df.to_csv(
        pred_file,
        index=False
    )

    print(f"\nSaved: {pred_file}")


    # ========================================================
    # Update summary
    # ========================================================

    new_result = pd.DataFrame([{
        "conformers": nconf,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson_r": corr,
        "runtime_min": runtime / 60
    }])

    # Remove an incomplete/old entry for this count if present
    summary = summary[
        summary["conformers"] != nconf
    ]

    summary = pd.concat(
        [summary, new_result],
        ignore_index=True
    )

    # Sort numerically by conformer number
    summary = summary.sort_values(
        "conformers"
    ).reset_index(drop=True)

    # Save immediately
    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        f"Updated: {SUMMARY_FILE}"
    )


# ============================================================
# Final summary
# ============================================================

summary = pd.read_csv(
    SUMMARY_FILE
)

summary = summary.sort_values(
    "conformers"
).reset_index(drop=True)

print("\n\n")
print("=" * 75)
print("ESOL — CONFORMER SCALING SUMMARY")
print("=" * 75)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved:")
print(SUMMARY_FILE)
