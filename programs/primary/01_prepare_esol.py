import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split

INPUT = "delaney-processed.csv"

df = pd.read_csv(INPUT)

# Keep only valid molecules and targets
df = df.dropna(
    subset=["smiles", "measured log solubility in mols per litre"]
).copy()

df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
df = df[df["mol"].notna()].copy()

# Flexibility descriptor
from rdkit.Chem import Lipinski
df["RotB"] = df["mol"].apply(Lipinski.NumRotatableBonds)

# Fixed 80/20 split
train, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

# Remove RDKit molecule object before saving
train = train.drop(columns=["mol"])
test = test.drop(columns=["mol"])

train.to_csv("esol_train.csv", index=False)
test.to_csv("esol_test.csv", index=False)

print("ESOL split created")
print("Training:", len(train))
print("Test:", len(test))
print("Train target mean:", train["measured log solubility in mols per litre"].mean())
print("Test target mean:", test["measured log solubility in mols per litre"].mean())
print("Train RotB mean:", train["RotB"].mean())
print("Test RotB mean:", test["RotB"].mean())
