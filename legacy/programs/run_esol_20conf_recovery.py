import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from qsarmil.modelling.lazy import (
    DESCRIPTORS,
    REGRESSORS,
    generate_conformers,
    calculate_descriptors,
    train_estimator,
    baseline_prediction,
)

from qsarmil.modelling.meta import GeneticSearch


# ============================================================
# ESOL — QSARmil 20-CONFORMER RECOVERY
#
# Recover an interrupted 20-conformer calculation.
#
# Existing model predictions in:
#     esol_qsarmil_20conf/
#
# are preserved.
#
# Only missing individual models are trained.
# ============================================================


TRAIN = "esol_train.csv"
TEST = "esol_test.csv"

OUTPUT_FOLDER = "esol_qsarmil_20conf"

FINAL_PREDICTION_FILE = "esol_qsarmil_20conf_predictions.csv"

SUMMARY_FILE = "esol_conformer_scaling_summary.csv"

NUM_CONF = 20
RANDOM_SEED = 42
NUM_CPU = 16
ACCELERATOR = "cpu"
HOPT = False


# ============================================================
# Load original ESOL train/test data
# ============================================================

print("\n")
print("=" * 80)
print("ESOL — QSARmil 20-CONFORMER RECOVERY")
print("=" * 80)


train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

smiles_train_all = train["smiles"].tolist()
smiles_test_all = test["smiles"].tolist()

y_train_all = train[
    "measured log solubility in mols per litre"
].tolist()

print(f"\nOriginal training molecules : {len(smiles_train_all)}")
print(f"Original test molecules     : {len(smiles_test_all)}")


# ============================================================
# Reproduce EXACT QSARmil train/validation split
# ============================================================

idx_train, idx_val = train_test_split(
    range(len(smiles_train_all)),
    test_size=0.2,
    random_state=RANDOM_SEED,
)

smi_train = [smiles_train_all[i] for i in idx_train]
y_train = [y_train_all[i] for i in idx_train]

smi_val = [smiles_train_all[i] for i in idx_val]
y_val = [y_train_all[i] for i in idx_val]

smi_test = list(smiles_test_all)

print(f"QSARmil training split        : {len(smi_train)}")
print(f"QSARmil validation split      : {len(smi_val)}")
print(f"QSARmil external test set     : {len(smi_test)}")


# ============================================================
# Check existing checkpoint files
# ============================================================

train_file = os.path.join(
    OUTPUT_FOLDER,
    "train.csv"
)

val_file = os.path.join(
    OUTPUT_FOLDER,
    "val.csv"
)

test_file = os.path.join(
    OUTPUT_FOLDER,
    "test.csv"
)

if not (
    os.path.exists(train_file)
    and os.path.exists(val_file)
    and os.path.exists(test_file)
):
    raise FileNotFoundError(
        "Existing QSARmil checkpoint files were not found."
    )


result_df_train = pd.read_csv(train_file)
result_df_val = pd.read_csv(val_file)
result_df_test = pd.read_csv(test_file)


# ============================================================
# Validate checkpoint structure
# ============================================================

print("\n")
print("-" * 80)
print("CHECKPOINT")
print("-" * 80)

print(
    f"train.csv : {result_df_train.shape}"
)

print(
    f"val.csv   : {result_df_val.shape}"
)

print(
    f"test.csv  : {result_df_test.shape}"
)


if list(result_df_train.columns) != list(result_df_val.columns):
    raise RuntimeError(
        "train.csv and val.csv do not have identical columns."
    )


expected_test_prefix = ["SMILES"]

if result_df_test.columns[0] != "SMILES":
    raise RuntimeError(
        "Unexpected first column in test.csv."
    )


# Existing model columns
existing_models = [
    c for c in result_df_test.columns
    if c != "SMILES"
]

print(
    f"\nExisting individual models : "
    f"{len(existing_models)}"
)


# ============================================================
# Expected complete model list
# ============================================================

expected_models = []

for desc_name in DESCRIPTORS.keys():

    for est_name in REGRESSORS.keys():

        expected_models.append(
            f"{desc_name}|{est_name}"
        )


print(
    f"Expected individual models  : "
    f"{len(expected_models)}"
)


# ============================================================
# Identify missing models
# ============================================================

missing_models = [
    model for model in expected_models
    if model not in existing_models
]


unexpected_models = [
    model for model in existing_models
    if model not in expected_models
]


if unexpected_models:

    print("\nWARNING — unexpected model columns:")
    for model in unexpected_models:
        print(" ", model)


print(
    f"\nMissing individual models   : "
    f"{len(missing_models)}"
)


if len(existing_models) + len(missing_models) != len(expected_models):

    raise RuntimeError(
        "Checkpoint model count is inconsistent."
    )


print("\nModels still requiring training:")

for i, model in enumerate(missing_models, 1):
    print(f"{i:2d}  {model}")


# ============================================================
# If everything is already complete
# ============================================================

if len(missing_models) == 0:

    print("\nAll 72 individual models already exist.")
    print("Skipping model training.")


# ============================================================
# Recreate 20-conformer molecular representation
# ============================================================

print("\n")
print("=" * 80)
print("STEP 1 — REGENERATING 20-CONFORMER REPRESENTATION")
print("=" * 80)


smi_all = (
    smi_train
    + smi_val
    + smi_test
)


conf_all = generate_conformers(
    smi_all,
    num_conf=NUM_CONF,
    num_cpu=NUM_CPU,
    verbose=True,
    random_seed=RANDOM_SEED,
)


n_train = len(smi_train)
n_val = len(smi_val)


conf_train_all = conf_all[
    :n_train
]

conf_val_all = conf_all[
    n_train:n_train + n_val
]

conf_test_all = conf_all[
    n_train + n_val:
]


# ============================================================
# Remove failed molecules exactly as LazyMIL does
# ============================================================

valid_idx_train = [
    i for i, c in enumerate(conf_train_all)
    if isinstance(c, list)
]

valid_idx_val = [
    i for i, c in enumerate(conf_val_all)
    if isinstance(c, list)
]

valid_idx_test = [
    i for i, c in enumerate(conf_test_all)
    if isinstance(c, list)
]


smi_train_valid = [
    smi_train[i]
    for i in valid_idx_train
]

y_train_valid = [
    y_train[i]
    for i in valid_idx_train
]

conf_train = [
    conf_train_all[i]
    for i in valid_idx_train
]


smi_val_valid = [
    smi_val[i]
    for i in valid_idx_val
]

y_val_valid = [
    y_val[i]
    for i in valid_idx_val
]

conf_val = [
    conf_val_all[i]
    for i in valid_idx_val
]


smi_test_valid = [
    smi_test[i]
    for i in valid_idx_test
]

conf_test = [
    conf_test_all[i]
    for i in valid_idx_test
]


train_baseline = baseline_prediction(
    y_train_valid,
    "continuous"
)


print(
    f"\nValid training molecules   : "
    f"{len(conf_train)}"
)

print(
    f"Valid validation molecules : "
    f"{len(conf_val)}"
)

print(
    f"Valid test molecules       : "
    f"{len(conf_test)}"
)


# ============================================================
# Calculate ONLY descriptors required by missing models
# ============================================================

missing_descriptors = []

for model_name in missing_models:

    desc_name = model_name.split("|")[0]

    if desc_name not in missing_descriptors:

        missing_descriptors.append(desc_name)


print("\n")
print("=" * 80)
print("STEP 2 — CALCULATING REQUIRED DESCRIPTORS")
print("=" * 80)

print("\nDescriptors required for recovery:")

for desc_name in missing_descriptors:
    print(" ", desc_name)


ready_descriptors = {}


for desc_name in missing_descriptors:

    print("\n")
    print(f"Calculating descriptor: {desc_name}")

    desc_factory = DESCRIPTORS[desc_name]

    desc_calc = desc_factory()

    x_all = calculate_descriptors(
        conf_train
        + conf_val
        + conf_test,
        desc_calc,
    )

    x_train = x_all[
        :len(conf_train)
    ]

    x_val = x_all[
        len(conf_train):
        len(conf_train) + len(conf_val)
    ]

    x_test = x_all[
        len(conf_train) + len(conf_val):
    ]

    ready_descriptors[desc_name] = (
        x_train,
        x_val,
        x_test,
    )

    print(
        f"{desc_name}: descriptor calculation complete"
    )


# ============================================================
# Prepare result dataframes
# ============================================================

# IMPORTANT:
# Load the existing checkpoint columns exactly as they are.
#
# We do NOT recreate the first 48 models.

# Existing checkpoint already contains:
#
# train: SMILES, Y_TRUE, 48 models
# val:   SMILES, Y_TRUE, 48 models
# test:  SMILES,       48 models


# ============================================================
# Recover missing models
# ============================================================

print("\n")
print("=" * 80)
print("STEP 3 — RECOVERING MISSING INDIVIDUAL MODELS")
print("=" * 80)

print(
    f"\nModels to train: {len(missing_models)}"
)


total_missing = len(missing_models)


for current_missing, model_name in enumerate(
    missing_models,
    1,
):

    print("\n")
    print("-" * 80)
    print(
        f"RECOVERY MODEL "
        f"{current_missing}/{total_missing}"
    )
    print(
        f"{model_name}"
    )
    print("-" * 80)


    desc_name, est_name = model_name.split(
        "|",
        1,
    )


    x_train, x_val, x_test = (
        ready_descriptors[desc_name]
    )


    est_factory = REGRESSORS[est_name]

    estimator = est_factory(
        accelerator=ACCELERATOR
    )


    start_model = time.time()


    pred_train, pred_val, pred_test = train_estimator(
        x_train,
        x_val,
        x_test,
        y_train_valid,
        y_val_valid,
        estimator,
        HOPT,
        random_seed=RANDOM_SEED,
        accelerator=ACCELERATOR,
    )


    # --------------------------------------------------------
    # Map predictions back to original molecule ordering
    # --------------------------------------------------------

    preds_train_by_smi = dict(
        zip(
            smi_train_valid,
            pred_train,
        )
    )

    preds_val_by_smi = dict(
        zip(
            smi_val_valid,
            pred_val,
        )
    )

    preds_test_by_smi = dict(
        zip(
            smi_test_valid,
            pred_test,
        )
    )


    result_df_train[model_name] = [
        preds_train_by_smi.get(
            smi,
            train_baseline,
        )
        for smi in smi_train
    ]


    result_df_val[model_name] = [
        preds_val_by_smi.get(
            smi,
            train_baseline,
        )
        for smi in smi_val
    ]


    result_df_test[model_name] = [
        preds_test_by_smi.get(
            smi,
            train_baseline,
        )
        for smi in smi_test
    ]


    # --------------------------------------------------------
    # SAVE IMMEDIATELY
    # --------------------------------------------------------

    result_df_train.to_csv(
        train_file,
        index=False,
    )

    result_df_val.to_csv(
        val_file,
        index=False,
    )

    result_df_test.to_csv(
        test_file,
        index=False,
    )


    runtime_model = (
        time.time()
        - start_model
    ) / 60.0


    print(
        f"\nCompleted: {model_name}"
    )

    print(
        f"Model runtime: "
        f"{runtime_model:.2f} min"
    )

    print(
        "Checkpoint saved."
    )

    print(
        f"Models now present: "
        f"{len(result_df_test.columns) - 1}/72"
    )


# ============================================================
# Verify that all 72 models exist
# ============================================================

result_df_train = pd.read_csv(
    train_file
)

result_df_val = pd.read_csv(
    val_file
)

result_df_test = pd.read_csv(
    test_file
)


final_models = [
    c for c in result_df_test.columns
    if c != "SMILES"
]


missing_after = [
    model for model in expected_models
    if model not in final_models
]


if missing_after:

    raise RuntimeError(
        "Recovery incomplete. Missing models:\n"
        + "\n".join(missing_after)
    )


if len(final_models) != 72:

    raise RuntimeError(
        f"Expected 72 models, found {len(final_models)}."
    )


print("\n")
print("=" * 80)
print("ALL 72 INDIVIDUAL MODELS ARE PRESENT")
print("=" * 80)


# ============================================================
# Step 4 — Genetic consensus
# ============================================================

print("\n")
print("=" * 80)
print("STEP 4 — GENETIC CONSENSUS SEARCH")
print("=" * 80)


x_val = result_df_val.iloc[:, 2:]

true_val = result_df_val.iloc[:, 1]


cons_search = GeneticSearch(
    cons_size="auto",
    n_iter=50,
)


best_consensus = cons_search.run(
    x_val,
    true_val,
)


best_consensus = list(
    best_consensus
)


print("\nBest genetic consensus:")

for name in best_consensus:

    print(
        f"  -{name}"
    )


# ============================================================
# Test prediction
# ============================================================

x_test = result_df_test.iloc[:, 1:]


missing_consensus_columns = [
    c for c in best_consensus
    if c not in x_test.columns
]


if missing_consensus_columns:

    raise RuntimeError(
        "Consensus references missing columns:\n"
        + "\n".join(
            missing_consensus_columns
        )
    )


prediction = list(
    cons_search.predict(
        x_test[
            best_consensus
        ]
    )
)


# ============================================================
# Save final prediction
# ============================================================

pred_df = test.copy()

pred_df["prediction"] = prediction

pred_df["num_conformers"] = NUM_CONF


pred_df.to_csv(
    FINAL_PREDICTION_FILE,
    index=False,
)


print("\n")
print(
    f"Saved: {FINAL_PREDICTION_FILE}"
)


# ============================================================
# Calculate final metrics
# ============================================================

y_test = test[
    "measured log solubility in mols per litre"
].values


r2 = r2_score(
    y_test,
    prediction,
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction,
    )
)


mae = mean_absolute_error(
    y_test,
    prediction,
)


corr = np.corrcoef(
    y_test,
    prediction,
)[0, 1]


print("\n")
print("=" * 80)
print("20-CONFORMER RECOVERY RESULT")
print("=" * 80)

print(
    f"N       = {len(y_test)}"
)

print(
    f"R²      = {r2:.6f}"
)

print(
    f"RMSE    = {rmse:.6f}"
)

print(
    f"MAE     = {mae:.6f}"
)

print(
    f"Pearson = {corr:.6f}"
)


# ============================================================
# Update scaling summary
# ============================================================

runtime_min = np.nan

new_result = pd.DataFrame(
    [{
        "conformers": NUM_CONF,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson_r": corr,
        "runtime_min": runtime_min,
    }]
)


if os.path.exists(SUMMARY_FILE):

    summary = pd.read_csv(
        SUMMARY_FILE
    )

    summary = summary[
        summary["conformers"] != NUM_CONF
    ]

else:

    summary = pd.DataFrame(
        columns=[
            "conformers",
            "R2",
            "RMSE",
            "MAE",
            "Pearson_r",
            "runtime_min",
        ]
    )


summary = pd.concat(
    [
        summary,
        new_result,
    ],
    ignore_index=True,
)


summary = summary.sort_values(
    "conformers"
).reset_index(drop=True)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print("\nUpdated:")
print(SUMMARY_FILE)


print("\n")
print("=" * 80)
print("FINAL SCALING SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\nRecovery completed successfully.")
