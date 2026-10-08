import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


TRAIN = "esol_train.csv"
TEST = "esol_test.csv"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)


# ------------------------------------------------------------
# RDKit 2D descriptor set
# ------------------------------------------------------------

descriptor_functions = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Crippen.MolLogP),
    ("TPSA", rdMolDescriptors.CalcTPSA),
    ("HBD", Lipinski.NumHDonors),
    ("HBA", Lipinski.NumHAcceptors),
    ("RotB", Lipinski.NumRotatableBonds),
    ("RingCount", Lipinski.RingCount),
    ("AromaticRings", Lipinski.NumAromaticRings),
    ("HeavyAtomCount", Lipinski.HeavyAtomCount),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
]


def calculate_descriptors(smiles_list):

    rows = []

    for smi in smiles_list:

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")

        row = {}

        for name, func in descriptor_functions:
            row[name] = func(mol)

        rows.append(row)

    return pd.DataFrame(rows)


print("=" * 70)
print("ESOL — SINGLE-INSTANCE (SIL) BASELINE")
print("=" * 70)

print(f"Training molecules = {len(train)}")
print(f"Test molecules     = {len(test)}")


X_train = calculate_descriptors(train["smiles"])
X_test = calculate_descriptors(test["smiles"])

y_train = train[
    "measured log solubility in mols per litre"
].values

y_test = test[
    "measured log solubility in mols per litre"
].values


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)


# ------------------------------------------------------------
# Overall performance
# ------------------------------------------------------------

r2 = r2_score(y_test, prediction)
rmse = np.sqrt(mean_squared_error(y_test, prediction))
mae = mean_absolute_error(y_test, prediction)
corr = np.corrcoef(y_test, prediction)[0, 1]


print("\nOverall performance")
print("-" * 70)

print(f"N    = {len(test)}")
print(f"R²   = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"r    = {corr:.4f}")


# ------------------------------------------------------------
# Flexibility
# ------------------------------------------------------------

test["prediction"] = prediction
test["RotB"] = test["smiles"].apply(
    lambda s: Lipinski.NumRotatableBonds(Chem.MolFromSmiles(s))
)

bins = [-1, 0, 3, 6, 100]
labels = ["0 RotB", "1–3 RotB", "4–6 RotB", "≥7 RotB"]

test["Flexibility"] = pd.cut(
    test["RotB"],
    bins=bins,
    labels=labels
)


print("\nFlexibility-stratified performance")
print("-" * 70)

for group, sub in test.groupby("Flexibility", observed=False):

    if len(sub) == 0:
        continue

    yy = sub[
        "measured log solubility in mols per litre"
    ]

    pp = sub["prediction"]

    print(
        f"{str(group):10s} "
        f"N={len(sub):3d}  "
        f"R²={r2_score(yy, pp):7.4f}  "
        f"RMSE={np.sqrt(mean_squared_error(yy, pp)):7.4f}  "
        f"MAE={mean_absolute_error(yy, pp):7.4f}"
    )


# ------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------

out = test[
    [
        "Compound ID",
        "smiles",
        "measured log solubility in mols per litre",
        "RotB",
        "Flexibility",
        "prediction",
    ]
]

out.to_csv(
    "esol_sil_predictions.csv",
    index=False
)

print("\nSaved:")
print("esol_sil_predictions.csv")
