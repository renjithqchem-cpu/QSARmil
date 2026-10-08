import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from rdkit import Chem
from qsarmil.modelling.lazy import generate_conformers
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.descriptor.rdkit import RDKitMORSE, RDKitGETAWAY
from qsarmil.descriptor.molfeat import MolFeatElectroShape, MolFeatPmapper

BASE = os.path.expanduser("~/Documents/qsarmil")
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "freesolv_descriptor_domain_analysis")
os.makedirs(OUT, exist_ok=True)

SEED = 42
NCONF = 5

benchmark = pd.read_csv(os.path.join(DATA, "freesolv_benchmark.csv"))

split = np.random.RandomState(SEED).permutation(len(benchmark))
n_train = int(0.8 * len(benchmark))
n_val = int(0.2 * n_train)

train_idx = split[:n_train]
test_idx = split[n_train:]

inner = np.random.RandomState(SEED).permutation(len(train_idx))
tr2_idx = train_idx[inner[:n_train-n_val]]
val_idx = train_idx[inner[n_train-n_val:]]

smiles = benchmark["smiles"].tolist()

def mols_from_smiles(indices):
    return [Chem.MolFromSmiles(smiles[i]) for i in indices]

all_mols = mols_from_smiles(range(len(smiles)))

print("Generating exact QSARmil conformers...")
conf_all = generate_conformers(all_mols, NCONF, SEED)

descriptor_specs = {
    "MORSE": RDKitMORSE(),
    "GETAWAY": RDKitGETAWAY(),
    "ElectroShape": MolFeatElectroShape(),
    "Pmapper": MolFeatPmapper(),
}

results = []
summary = []

def euclidean_spread(X):
    if len(X) < 2:
        return 0.0
    mu = X.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((X - mu) ** 2, axis=1))))

for name, transformer in descriptor_specs.items():

    print("\n==============================")
    print(name)
    print("==============================")

    wrapper = DescriptorWrapper(transformer, verbose=False)
    bags = wrapper.run(conf_all)

    valid = [isinstance(x, np.ndarray) for x in bags]
    print("Valid:", sum(valid), "/", len(bags))

    if not all(valid):
        failed = np.where(~np.array(valid))[0]
        print("Failed molecules:", len(failed))

    valid_bags = [x for x in bags if isinstance(x, np.ndarray)]

    stacked_train = np.vstack(
        [bags[i] for i in tr2_idx if isinstance(bags[i], np.ndarray)]
    ).astype(float)

    # Train-only standardization
    mu = stacked_train.mean(axis=0)
    sd = stacked_train.std(axis=0)
    sd[sd == 0] = 1.0

    scaled_bags = []
    for b in bags:
        if isinstance(b, np.ndarray):
            scaled_bags.append((b - mu) / sd)
        else:
            scaled_bags.append(None)

    train_centroids = []
    train_ids = []

    for i in tr2_idx:
        b = scaled_bags[i]
        if b is not None:
            train_centroids.append(b.mean(axis=0))
            train_ids.append(i)

    train_centroids = np.asarray(train_centroids)

    for i, b in enumerate(scaled_bags):

        if b is None:
            continue

        centroid = b.mean(axis=0)
        spread = euclidean_spread(b)

        distances = np.linalg.norm(train_centroids - centroid, axis=1)

        nearest = float(np.min(distances))
        mean_dist = float(np.mean(distances))

        results.append({
            "index": i,
            "descriptor": name,
            "split": (
                "train" if i in set(tr2_idx)
                else "validation" if i in set(val_idx)
                else "test"
            ),
            "n_conformers": len(b),
            "descriptor_dimension": b.shape[1],
            "ensemble_spread": spread,
            "nearest_train_distance": nearest,
            "mean_train_distance": mean_dist,
        })

    print("Descriptor dimension:", stacked_train.shape[1])

df = pd.DataFrame(results)

df.to_csv(
    os.path.join(OUT, "freesolv_descriptor_domain_metrics.csv"),
    index=False
)

print("\nSaved:")
print(os.path.join(OUT, "freesolv_descriptor_domain_metrics.csv"))

# ---------------------------------------------------------
# Merge with existing 2D-SIL and 5-conf MIL predictions
# ---------------------------------------------------------

predfile = os.path.join(DATA, "freesolv_2d_sil_predictions.csv")

if os.path.exists(predfile):
    pred = pd.read_csv(predfile)

    print("\n2D prediction columns:")
    print(pred.columns.tolist())

    if "y_true" in pred.columns:
        ycol = "y_true"
    elif "experimental_dG_hyd_kcal_mol" in pred.columns:
        ycol = "experimental_dG_hyd_kcal_mol"
    else:
        ycol = None

    if ycol is not None:
        pred["err_2d"] = (
            pred[ycol] - pred["prediction"]
        ).abs()

        pred["index"] = np.arange(len(pred))

        df2 = df[df["split"] == "test"].copy()

        merged = df2.merge(
            pred[["index", "err_2d"]],
            on="index",
            how="left"
        )

        merged.to_csv(
            os.path.join(
                OUT,
                "freesolv_test_domain_vs_2d_error.csv"
            ),
            index=False
        )

        for desc in descriptor_specs:

            x = merged[
                merged["descriptor"] == desc
            ]

            for metric in [
                "ensemble_spread",
                "nearest_train_distance",
                "mean_train_distance"
            ]:

                z = x[[metric, "err_2d"]].dropna()

                if len(z) > 5:
                    rho, p = spearmanr(
                        z[metric],
                        z["err_2d"]
                    )

                    summary.append({
                        "descriptor": desc,
                        "metric": metric,
                        "n": len(z),
                        "spearman_rho": rho,
                        "p_value": p
                    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    os.path.join(
        OUT,
        "freesolv_domain_error_correlations.csv"
    ),
    index=False
)

print("\nCorrelation summary:")
print(summary_df.to_string(index=False))

print("\nDONE.")
