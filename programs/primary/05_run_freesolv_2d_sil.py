import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr


TRAIN_FILE = "data/freesolv_train_seed42.csv"
TEST_FILE = "data/freesolv_test_seed42.csv"

OUTPUT_FILE = "data/freesolv_2d_sil_predictions.csv"


def calculate_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.RingCount(mol),
        Lipinski.NumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ]


FEATURE_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotB",
    "RingCount",
    "AromaticRings",
    "HeavyAtomCount",
    "FractionCSP3",
]


def metrics(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r = pearsonr(y_true, y_pred)[0]
    rho = spearmanr(y_true, y_pred)[0]

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson_r": r,
        "Spearman_rho": rho,
    }


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print("=" * 75)
print("FreeSolv 2D-SIL BASELINE")
print("=" * 75)

print("Train:", len(train))
print("Test :", len(test))

# ---------------------------------------------------------
# Descriptors
# ---------------------------------------------------------

X_train = np.array([
    calculate_descriptors(s)
    for s in train["smiles"]
])

X_test = np.array([
    calculate_descriptors(s)
    for s in test["smiles"]
])

y_train = train["experimental_dG_hyd_kcal_mol"].values
y_test = test["experimental_dG_hyd_kcal_mol"].values

print("\nDescriptor matrix:")
print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

m = metrics(y_test, pred)

print("\nOverall performance:")
for k, v in m.items():
    print(f"{k:15s}: {v:.6f}")

# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------

out = test[
    [
        "freesolv_id",
        "smiles",
        "experimental_dG_hyd_kcal_mol"
    ]
].copy()

out["prediction"] = pred
out["error"] = pred - y_test
out["absolute_error"] = np.abs(pred - y_test)

for name, value in m.items():
    out.attrs[name] = value

out.to_csv(OUTPUT_FILE, index=False)

# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

importance = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print("\nFeature importance:")
print(importance.to_string(index=False))

importance.to_csv(
    "data/freesolv_2d_sil_feature_importance.csv",
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)
print("data/freesolv_2d_sil_feature_importance.csv")

print("=" * 75)
