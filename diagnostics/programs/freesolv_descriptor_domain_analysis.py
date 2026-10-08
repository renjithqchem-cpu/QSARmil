import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem
from qsarmil.conformer.rdkit import RDKitConformerGenerator
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.descriptor.rdkit import RDKitMORSE, RDKitGETAWAY
from molfeat.calc import ElectroShapeDescriptors, Pharmacophore3D

BASE = os.path.expanduser("~/Documents/qsarmil")
DATA = os.path.join(BASE, "data")
PROD = os.path.join(BASE, "freesolv_qsarmil_5conf")
OUT = os.path.join(BASE, "freesolv_descriptor_domain_analysis")
os.makedirs(OUT, exist_ok=True)

NCONF = 5
SEED = 42

train = pd.read_csv(os.path.join(PROD, "train.csv"))
val = pd.read_csv(os.path.join(PROD, "val.csv"))
test = pd.read_csv(os.path.join(PROD, "test.csv"))
pred2d = pd.read_csv(os.path.join(DATA, "freesolv_2d_sil_predictions.csv"))

train["split"] = "train"
val["split"] = "validation"
test["split"] = "test"

all_split = pd.concat(
    [
        train[["SMILES", "split"]],
        val[["SMILES", "split"]],
        test[["SMILES", "split"]],
    ],
    ignore_index=True,
)

print("Production split sizes:")
print(all_split["split"].value_counts())
print("Total:", len(all_split))

# Preserve exact production order.
smiles = all_split["SMILES"].tolist()
mols = [Chem.MolFromSmiles(s) for s in smiles]

generator = RDKitConformerGenerator(
    num_conf=NCONF,
    e_thresh=None,
    random_seed=SEED,
    verbose=True,
)

print("\nGenerating exact QSARmil conformers...")
conf_bags = generator.run(mols)

descriptor_specs = {
    "MORSE": RDKitMORSE(),
    "GETAWAY": RDKitGETAWAY(),
    "ElectroShape": ElectroShapeDescriptors(),
    "Pmapper": Pharmacophore3D(factory="pmapper"),
}

split_array = np.array(all_split["split"].tolist())

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def centroid_and_spread(bag):
    centroid = np.mean(bag, axis=0)
    spread = float(
        np.sqrt(
            np.mean(
                np.sum((bag - centroid) ** 2, axis=1)
            )
        )
    )
    return centroid, spread


rows = []
correlation_rows = []

for name, transformer in descriptor_specs.items():

    print("\n===================================")
    print(name)
    print("===================================")

    wrapper = DescriptorWrapper(
        transformer,
        verbose=False
    )

    bags = wrapper.run(conf_bags)

    valid = np.array(
        [isinstance(x, np.ndarray) for x in bags]
    )

    print(
        "Valid descriptor bags:",
        int(valid.sum()),
        "/",
        len(valid)
    )

    if not valid.all():
        print(
            "Failed:",
            int((~valid).sum())
        )

    # --------------------------------------------------------
    # EXACT QSARmil-style global invalid/extreme-column filter
    # --------------------------------------------------------

    valid_bags = [
        b for b in bags
        if isinstance(b, np.ndarray)
    ]

    stacked = np.vstack(valid_bags).astype(float)

    stacked[np.abs(stacked) >= 1e25] = np.nan

    keep_mask = ~np.isnan(stacked).any(axis=0)

    print(
        "Original dimensions:",
        len(keep_mask)
    )
    print(
        "Removed columns:",
        int((~keep_mask).sum())
    )
    print(
        "Final dimensions:",
        int(keep_mask.sum())
    )

    cleaned = []

    for b in bags:

        if not isinstance(b, np.ndarray):
            cleaned.append(None)
            continue

        b = b.astype(float)
        b[np.abs(b) >= 1e25] = np.nan
        b = b[:, keep_mask]

        cleaned.append(b)

    # --------------------------------------------------------
    # TRAIN-ONLY scaling
    #
    # This deliberately uses train.csv + val.csv because
    # those are the 513 molecules used to fit the final
    # QSARmil models before independent test prediction.
    # --------------------------------------------------------

    final_train_mask = (
        (split_array == "train") |
        (split_array == "validation")
    )

    train_bags = [
        cleaned[i]
        for i in range(len(cleaned))
        if final_train_mask[i] and cleaned[i] is not None
    ]

    train_stack = np.vstack(train_bags)

    scaler_mean = train_stack.mean(axis=0)
    scaler_std = train_stack.std(axis=0)

    scaler_std[scaler_std == 0] = 1.0

    scaled = []

    for b in cleaned:

        if b is None:
            scaled.append(None)
        else:
            scaled.append(
                (b - scaler_mean) / scaler_std
            )

    # --------------------------------------------------------
    # Training molecular centroids
    # --------------------------------------------------------

    train_centroids = []

    for i in range(len(scaled)):

        if split_array[i] not in ("train", "validation"):
            continue

        if scaled[i] is None:
            continue

        centroid, _ = centroid_and_spread(
            scaled[i]
        )

        train_centroids.append(centroid)

    train_centroids = np.asarray(train_centroids)

    print(
        "Training reference molecules:",
        len(train_centroids)
    )

    # --------------------------------------------------------
    # Molecule-level metrics
    # --------------------------------------------------------

    for i, bag in enumerate(scaled):

        if bag is None:
            continue

        centroid, spread = centroid_and_spread(bag)

        distances = np.linalg.norm(
            train_centroids - centroid,
            axis=1
        )

        rows.append(
            {
                "row_index": i,
                "SMILES": smiles[i],
                "split": split_array[i],
                "descriptor": name,
                "n_conformers": bag.shape[0],
                "descriptor_dimension": bag.shape[1],
                "ensemble_spread": spread,
                "nearest_train_distance": float(
                    np.min(distances)
                ),
                "mean_train_distance": float(
                    np.mean(distances)
                ),
            }
        )

domain = pd.DataFrame(rows)

domain_file = os.path.join(
    OUT,
    "freesolv_descriptor_domain_metrics.csv"
)

domain.to_csv(domain_file, index=False)

print("\nSaved:")
print(domain_file)

# ------------------------------------------------------------
# Merge test-domain metrics with 2D-SIL errors
# ------------------------------------------------------------

test_domain = domain[
    domain["split"] == "test"
].copy()

# Match 2D predictions by SMILES.
merged = test_domain.merge(
    pred2d[
        [
            "smiles",
            "experimental_dG_hyd_kcal_mol",
            "prediction",
            "absolute_error",
        ]
    ],
    left_on="SMILES",
    right_on="smiles",
    how="left",
)

merged["mil_improvement_needed"] = np.nan

# ------------------------------------------------------------
# Calculate errors of selected individual MIL models
# ------------------------------------------------------------

model_files = {
    "MORSE_MeanInstanceWrapper":
        "RDKitMORSE|MeanInstanceWrapperMLPNetworkRegressor",

    "Pmapper_MeanInstanceWrapper":
        "MolFeatPmapper|MeanInstanceWrapperMLPNetworkRegressor",

    "ElectroShape_MeanInstanceWrapper":
        "MolFeatElectroShape|MeanInstanceWrapperMLPNetworkRegressor",

    "GETAWAY_MeanInstanceWrapper":
        "RDKitGETAWAY|MeanInstanceWrapperMLPNetworkRegressor",
}

test_models = test.drop(
    columns=["split"],
    errors="ignore"
)

for short, col in model_files.items():

    if col not in test_models.columns:
        print("Missing:", col)
        continue

    pred_col = test_models[col].to_numpy()

    # test.csv and all_split have identical production order.
    y_true = np.array(
        [
            pred2d.loc[
                pred2d["smiles"] == s,
                "experimental_dG_hyd_kcal_mol"
            ].iloc[0]
            for s in test["SMILES"]
        ]
    )

    abs_err = np.abs(
        y_true - pred_col
    )

    error_df = pd.DataFrame(
        {
            "SMILES": test["SMILES"],
            short + "_absolute_error": abs_err,
            short + "_prediction": pred_col,
        }
    )

    merged = merged.merge(
        error_df,
        on="SMILES",
        how="left",
    )

# ------------------------------------------------------------
# Improvement relative to 2D SIL
# positive = MIL error smaller
# ------------------------------------------------------------

for short in model_files:

    err_col = short + "_absolute_error"

    if err_col in merged.columns:

        merged[
            short + "_improvement"
        ] = (
            merged["absolute_error"]
            - merged[err_col]
        )

merged_file = os.path.join(
    OUT,
    "freesolv_test_domain_vs_model_errors.csv"
)

merged.to_csv(
    merged_file,
    index=False
)

print("\nSaved:")
print(merged_file)

# ------------------------------------------------------------
# Correlations
# ------------------------------------------------------------

metrics = [
    "ensemble_spread",
    "nearest_train_distance",
    "mean_train_distance",
]

for descriptor in descriptor_specs:

    sub = merged[
        merged["descriptor"] == descriptor
    ].copy()

    for metric in metrics:

        for short in model_files:

            improvement_col = (
                short + "_improvement"
            )

            if improvement_col not in sub.columns:
                continue

            z = sub[
                [metric, improvement_col]
            ].dropna()

            if len(z) < 10:
                continue

            rho, p = spearmanr(
                z[metric],
                z[improvement_col]
            )

            correlation_rows.append(
                {
                    "descriptor": descriptor,
                    "metric": metric,
                    "model": short,
                    "n": len(z),
                    "spearman_rho": rho,
                    "p_value": p,
                }
            )

corr = pd.DataFrame(
    correlation_rows
)

corr_file = os.path.join(
    OUT,
    "freesolv_domain_vs_mil_improvement_correlations.csv"
)

corr.to_csv(
    corr_file,
    index=False
)

print("\nCorrelation results:")
print(
    corr.sort_values(
        ["descriptor", "metric", "p_value"]
    ).to_string(index=False)
)

print("\nSaved:")
print(corr_file)

# ------------------------------------------------------------
# Distribution summary
# ------------------------------------------------------------

summary = (
    domain
    .groupby(["descriptor", "split"])
    [
        [
            "ensemble_spread",
            "nearest_train_distance",
            "mean_train_distance",
        ]
    ]
    .agg(["mean", "median", "std"])
)

summary_file = os.path.join(
    OUT,
    "freesolv_descriptor_domain_summary.csv"
)

summary.to_csv(summary_file)

print("\nDistribution summary:")
print(summary)

print("\n===================================")
print("ANALYSIS COMPLETE")
print("===================================")
