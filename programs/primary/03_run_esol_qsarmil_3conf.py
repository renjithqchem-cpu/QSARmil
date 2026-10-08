import pandas as pd
import numpy as np
from qsarmil.modelling.meta import MultiConformerRegressor

TRAIN = "esol_train.csv"
TEST = "esol_test.csv"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

smiles_train = train["smiles"].tolist()
y_train = train["measured log solubility in mols per litre"].to_numpy()

smiles_test = test["smiles"].tolist()

print("Training molecules:", len(smiles_train))
print("Test molecules:", len(smiles_test))

model = MultiConformerRegressor(
    num_conf=3,
    hopt=False,
    verbose=True
)

y_pred = model.train_predict(
    smiles_train,
    y_train,
    smiles_test
)

# Save predictions
out = test[["Compound ID", "smiles",
            "measured log solubility in mols per litre",
            "RotB"]].copy()

out["prediction"] = np.asarray(y_pred)
out["residual"] = (
    out["prediction"]
    - out["measured log solubility in mols per litre"]
)

out.to_csv("esol_qsarmil_3conf_predictions.csv", index=False)

print("\nSaved:")
print("esol_qsarmil_3conf_predictions.csv")

print("\nPrediction summary:")
print(out["prediction"].describe())
