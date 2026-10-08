import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem

val = pd.read_csv("freesolv_qsarmil_5conf/val.csv")
test = pd.read_csv("freesolv_qsarmil_5conf/test.csv")
bench = pd.read_csv("data/freesolv_benchmark.csv")

models = [c for c in val.columns if c not in ["SMILES", "Y_TRUE"]]

def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)

test["CANON"] = test["SMILES"].map(canon)
bench["CANON"] = bench["smiles"].map(canon)

target_map = bench.set_index("CANON")["experimental_dG_hyd_kcal_mol"]
test["Y_TRUE"] = test["CANON"].map(target_map)

if test["Y_TRUE"].isna().any():
    raise RuntimeError("Test molecules could not all be matched to FreeSolv.")

rows = []

for m in models:
    yv = val["Y_TRUE"]
    pv = val[m]
    yt = test["Y_TRUE"]
    pt = test[m]

    rows.append({
        "model": m,
        "val_R2": r2_score(yv, pv),
        "test_R2": r2_score(yt, pt),
        "val_RMSE": np.sqrt(mean_squared_error(yv, pv)),
        "test_RMSE": np.sqrt(mean_squared_error(yt, pt)),
        "val_MAE": mean_absolute_error(yv, pv),
        "test_MAE": mean_absolute_error(yt, pt)
    })

a = pd.DataFrame(rows)

a["val_rank"] = a["val_RMSE"].rank()
a["test_rank"] = a["test_RMSE"].rank()
a["rank_shift"] = a["test_rank"] - a["val_rank"]

print("=" * 100)
print("FREESOLV 72-MODEL VALIDATION / TEST FORENSIC AUDIT")
print("=" * 100)

print("Validation molecules:", len(val))
print("Test molecules:", len(test))
print("Models:", len(models))

r, p = pearsonr(a["val_RMSE"], a["test_RMSE"])
rho, p2 = spearmanr(a["val_RMSE"], a["test_RMSE"])

r2r, p3 = pearsonr(a["val_R2"], a["test_R2"])
r2rho, p4 = spearmanr(a["val_R2"], a["test_R2"])

print()
print("VALIDATION -> TEST STABILITY")
print("RMSE Pearson:", r, "p =", p)
print("RMSE Spearman:", rho, "p =", p2)
print("R2 Pearson:", r2r, "p =", p3)
print("R2 Spearman:", r2rho, "p =", p4)

print()
print("TOP 15 BY VALIDATION RMSE")
print(
    a.sort_values("val_RMSE").head(15).to_string(index=False)
)

print()
print("TOP 15 BY TEST RMSE")
print(
    a.sort_values("test_RMSE").head(15).to_string(index=False)
)

selected4 = [
    "MolFeatElectroShape|MeanInstanceWrapperMLPNetworkRegressor",
    "RDKitMORSE|MeanInstanceWrapperMLPNetworkRegressor",
    "MolFeatPmapper|MeanInstanceWrapperMLPNetworkRegressor",
    "RDKitGETAWAY|MeanInstanceWrapperMLPNetworkRegressor"
]

selected2 = [
    "RDKitMORSE|MeanInstanceWrapperMLPNetworkRegressor",
    "MolFeatPmapper|MeanInstanceWrapperMLPNetworkRegressor"
]

print()
print("QSARCONS SELECTED MODELS")
print(
    a[a["model"].isin(selected4)].to_string(index=False)
)

def report(name, cols):
    pv = val[cols].mean(axis=1)
    pt = test[cols].mean(axis=1)

    print()
    print(name)
    print(
        "Validation R2:",
        r2_score(val["Y_TRUE"], pv),
        "RMSE:",
        np.sqrt(mean_squared_error(val["Y_TRUE"], pv)),
        "MAE:",
        mean_absolute_error(val["Y_TRUE"], pv)
    )
    print(
        "Test R2:",
        r2_score(test["Y_TRUE"], pt),
        "RMSE:",
        np.sqrt(mean_squared_error(test["Y_TRUE"], pt)),
        "MAE:",
        mean_absolute_error(test["Y_TRUE"], pt)
    )

print()
print("CONSENSUS RESULTS")
report("k=2", selected2)
report("k=4", selected4)

print()
print("LEAVE-ONE-MEMBER-OUT k=4")

for removed in selected4:
    remaining = [x for x in selected4 if x != removed]
    report("Removed: " + removed, remaining)

print()
print("TEST ERROR CORRELATIONS")

errors = pd.DataFrame()

for m in selected4:
    errors[m] = test["Y_TRUE"] - test[m]

print(errors.corr().round(4).to_string())

print()
print("SELECTED MODEL PREDICTION RANGES")

for m in selected4:
    print(m)
    print(
        "validation:",
        val[m].min(),
        val[m].max(),
        val[m].mean()
    )
    print(
        "test:",
        test[m].min(),
        test[m].max(),
        test[m].mean()
    )

outfile = "freesolv_qsarmil_5conf/model_validation_test_forensics.csv"
a.to_csv(outfile, index=False)

print()
print("Saved:", outfile)
print("AUDIT COMPLETE")
