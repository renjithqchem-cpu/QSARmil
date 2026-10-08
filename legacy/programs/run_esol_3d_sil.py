import os
import sys
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# QSARmil descriptor machinery
from qsarmil.modelling import lazy


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ESOL_FILE = os.path.join(BASE, "data", "delaney-processed.csv")
OUT_DIR = os.path.join(BASE, "data", "esol_3d_sil")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_CONFS = 20


def generate_lowest_uff_conformer(smiles):
    """
    Reproduce the QSARmil RDKit conformer protocol:
      AddHs -> ETKDGv3(seed=42) -> EmbedMultipleConfs(20)
      -> UFF optimization -> select lowest UFF-energy conformer.

    Returns a single conformer-containing molecule with one conformer.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = SEED

    ids = list(
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=N_CONFS,
            params=params
        )
    )

    if not ids:
        return None

    energies = []

    for cid in ids:
        try:
            status = AllChem.UFFOptimizeMolecule(
                mol,
                confId=cid
            )

            ff = AllChem.UFFGetMoleculeForceField(
                mol,
                confId=cid
            )

            energy = ff.CalcEnergy()

            if np.isfinite(energy):
                energies.append((float(energy), cid))

        except Exception:
            continue

    if not energies:
        return None

    energies.sort(key=lambda x: x[0])
    lowest_energy, best_cid = energies[0]

    # Keep exactly the selected conformer.
    out = Chem.Mol(mol)
    out.RemoveAllConformers()

    conf = mol.GetConformer(best_cid)
    out.AddConformer(Chem.Conformer(conf), assignId=True)

    return out, lowest_energy, len(ids)


def calculate_descriptor(mol, desc_name):
    """
    Use the same QSARmil descriptor factory used in the MIL calculations.
    """
    factory = lazy.DESCRIPTORS[desc_name]
    descriptor = factory()

    # QSARmil calculate_descriptors expects a list of conformer ensembles.
    result = lazy.calculate_descriptors([[mol]], descriptor)

    # result[0] corresponds to the one-molecule ensemble.
    x = result[0]

    x = np.asarray(x, dtype=float)

    if x.ndim == 2:
        x = x[0]

    return x


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    return {
        "N": len(y),
        "R2": r2_score(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAE": mean_absolute_error(y, pred),
        "Pearson_r": pearsonr(y, pred)[0],
    }


print("=" * 80)
print("ESOL — 3D SINGLE-INSTANCE BASELINE")
print("=" * 80)

esol = pd.read_csv(ESOL_FILE)

# Identify columns exactly as used previously.
SMILES_COL = "smiles"
TARGET_COL = "measured log solubility in mols per litre"
ROTB_COL = "Number of Rotatable Bonds"

smiles = esol[SMILES_COL].astype(str).tolist()
y = esol[TARGET_COL].astype(float).to_numpy()
rotb = esol[ROTB_COL].astype(int).to_numpy()

idx = np.arange(len(esol))

train_idx, test_idx = train_test_split(
    idx,
    test_size=0.20,
    random_state=SEED
)

print(f"Dataset molecules = {len(esol)}")
print(f"Training molecules = {len(train_idx)}")
print(f"Test molecules     = {len(test_idx)}")
print()

# ----------------------------------------------------------------------
# Generate exactly one representative conformer for every molecule.
# ----------------------------------------------------------------------

print("Generating 20 conformers and selecting lowest-UFF-energy conformer...")
print()

molecules = []
energies = []
n_generated = []

failed = []

for i, smi in enumerate(smiles):

    result = generate_lowest_uff_conformer(smi)

    if result is None:
        molecules.append(None)
        energies.append(np.nan)
        n_generated.append(0)
        failed.append(i)
    else:
        mol, energy, nconf = result
        molecules.append(mol)
        energies.append(energy)
        n_generated.append(nconf)

    if (i + 1) % 100 == 0:
        print(f"{i+1}/{len(smiles)} processed")

print()
print(f"Successful molecules = {len(esol) - len(failed)}")
print(f"Failed molecules     = {len(failed)}")

if failed:
    print("WARNING: failed indices:", failed)

# ----------------------------------------------------------------------
# Descriptors
# ----------------------------------------------------------------------

descriptor_names = list(lazy.DESCRIPTORS.keys())

print()
print("3D descriptors:")
for x in descriptor_names:
    print(" ", x)

# Store descriptor matrices.
descriptor_data = {}

valid = [i for i, m in enumerate(molecules) if m is not None]

for desc_name in descriptor_names:

    print()
    print(f"Calculating {desc_name}...")

    X = []

    for i in valid:
        try:
            X.append(calculate_descriptor(molecules[i], desc_name))
        except Exception as exc:
            print(
                f"Descriptor failure: {desc_name}, molecule {i}: {exc}"
            )
            X.append(None)

    # Determine dimensionality from valid first result.
    first = next((x for x in X if x is not None), None)

    if first is None:
        print(f"  FAILED: no descriptors generated")
        continue

    n_features = len(first)

    X_full = np.full(
        (len(esol), n_features),
        np.nan,
        dtype=float
    )

    for i, x in zip(valid, X):
        if x is not None and len(x) == n_features:
            X_full[i] = x

    # Remove non-finite descriptor columns.
    X_train_raw = X_full[train_idx]

    valid_feature_cols = np.any(
        np.isfinite(X_train_raw),
        axis=0
    )

    removed = np.sum(~valid_feature_cols)

    X_full = X_full[:, valid_feature_cols]

    X_train_raw = X_full[train_idx]

    medians = np.nanmedian(
        X_train_raw,
        axis=0
    )

    usable_cols = np.isfinite(medians)

    removed += np.sum(~usable_cols)

    X_full = X_full[:, usable_cols]
    medians = medians[usable_cols]

    bad = ~np.isfinite(X_full)

    rows, cols = np.where(bad)

    X_full[rows, cols] = medians[cols]

    descriptor_data[desc_name] = X_full

    print(
        f"  features = {X_full.shape[1]}"
        f" | removed non-finite columns = {removed}"
    )

# ----------------------------------------------------------------------
# Train conventional single-instance RF models.
#
# This provides a genuine 3D-SIL baseline:
# one conformer -> one descriptor vector -> one prediction.
# ----------------------------------------------------------------------

predictions = pd.DataFrame({
    "SMILES": smiles,
    "Y_TRUE": y,
    "RotB": rotb,
    "UFF_Energy": energies,
    "N_Conformers_Generated": n_generated,
})

results = []

for desc_name, X in descriptor_data.items():

    print()
    print("=" * 80)
    print(desc_name)
    print("=" * 80)

    X_train = X[train_idx]
    X_test = X[test_idx]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=SEED,
        n_jobs=-1,
        max_features="sqrt"
    )

    model.fit(
        X_train,
        y[train_idx]
    )

    pred_test = model.predict(X_test)

    pred_full = np.full(len(esol), np.nan)
    pred_full[test_idx] = pred_test

    predictions[desc_name] = pred_full

    m = metrics(
        y[test_idx],
        pred_test
    )

    m["Descriptor"] = desc_name

    results.append(m)

    print(
        f"N    = {m['N']}\n"
        f"R²   = {m['R2']:.6f}\n"
        f"RMSE = {m['RMSE']:.6f}\n"
        f"MAE  = {m['MAE']:.6f}\n"
        f"r    = {m['Pearson_r']:.6f}"
    )

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df[
    ["Descriptor", "N", "R2", "RMSE", "MAE", "Pearson_r"]
]

results_df.to_csv(
    os.path.join(
        OUT_DIR,
        "esol_3d_sil_descriptor_metrics.csv"
    ),
    index=False
)

predictions.to_csv(
    os.path.join(
        OUT_DIR,
        "esol_3d_sil_predictions.csv"
    ),
    index=False
)

# Save selected conformer information.
conf_info = pd.DataFrame({
    "row": np.arange(len(esol)),
    "SMILES": smiles,
    "UFF_Energy": energies,
    "N_Conformers_Generated": n_generated,
})

conf_info.to_csv(
    os.path.join(
        OUT_DIR,
        "selected_conformer_info.csv"
    ),
    index=False
)

print()
print("=" * 80)
print("3D-SIL COMPLETE")
print("=" * 80)

print()
print(results_df.to_string(index=False))

print()
print("Saved:")
print(
    os.path.join(
        OUT_DIR,
        "esol_3d_sil_descriptor_metrics.csv"
    )
)
print(
    os.path.join(
        OUT_DIR,
        "esol_3d_sil_predictions.csv"
    )
)
print(
    os.path.join(
        OUT_DIR,
        "selected_conformer_info.csv"
    )
)
