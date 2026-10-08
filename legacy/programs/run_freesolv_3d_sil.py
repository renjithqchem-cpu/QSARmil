import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from qsarmil.modelling import lazy

DATA_FILE = "data/freesolv_benchmark.csv"
TRAIN_FILE = "data/freesolv_train_seed42.csv"
TEST_FILE = "data/freesolv_test_seed42.csv"
TRAIN_FILE = "data/freesolv_train_seed42.csv"
TEST_FILE = "data/freesolv_test_seed42.csv"
OUTPUT_DIR = "data/freesolv_3d_sil"
SEED = 42
N_CONFORMERS = 20


def generate_lowest_energy_conformer(smiles, num_conf=20, seed=42):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, np.nan, np.nan, -1
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conf, params=params))
    if not conf_ids:
        return None, np.nan, np.nan, -1
    energies = []
    for cid in conf_ids:
        try:
            AllChem.UFFOptimizeMolecule(mol_h, confId=cid, maxIters=2000)
            ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)
            e = float(ff.CalcEnergy())
            if np.isfinite(e):
                energies.append((cid, e))
        except Exception:
            pass
    if not energies:
        return None, np.nan, np.nan, -1
    energies.sort(key=lambda x: x[1])
    best_cid, best_e = energies[0]
    selected = Chem.Mol(mol_h)
    selected.RemoveAllConformers()
    selected.AddConformer(mol_h.GetConformer(best_cid), assignId=True)
    selected = Chem.RemoveHs(selected)
    return selected, best_e, energies[-1][1], best_cid


def pearson(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
TARGET = "experimental_dG_hyd_kcal_mol"
smiles_all = df["smiles"].astype(str).tolist()
y_all = df[TARGET].astype(float).tolist()
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

train_smiles_set = set(train_df["smiles"].astype(str))
test_smiles_set = set(test_df["smiles"].astype(str))

train_idx = [
    i for i, smi in enumerate(smiles_all)
    if smi in train_smiles_set
]

test_idx = [
    i for i, smi in enumerate(smiles_all)
    if smi in test_smiles_set
]

assert len(train_idx) == 513
assert len(test_idx) == 129
assert train_smiles_set.isdisjoint(test_smiles_set)

print("=" * 80)
print("FreeSolv 3D-SIL NEURAL BENCHMARK")
print("=" * 80)
print(f"Total molecules = {len(smiles_all)}")
print(f"Original train = {len(train_idx)}; original test = {len(test_idx)}")

molecules = []
energies = []
max_energies = []
conf_ids = []
rotb = []
failed = []

for i, smi in enumerate(smiles_all):
    mol, emin, emax, cid = generate_lowest_energy_conformer(smi, N_CONFORMERS, SEED)
    molecules.append(mol)
    energies.append(emin)
    max_energies.append(emax)
    conf_ids.append(cid)
    om = Chem.MolFromSmiles(smi)
    rotb.append(np.nan if om is None else int(Lipinski.NumRotatableBonds(om)))
    if mol is None:
        failed.append(i)
    if (i + 1) % 25 == 0 or i == len(smiles_all) - 1:
        print(f"{i + 1}/{len(smiles_all)} processed")

valid = [i for i, m in enumerate(molecules) if m is not None]
train_valid = [i for i in valid if i in set(train_idx)]
test_valid = [i for i in valid if i in set(test_idx)]

smiles_train = [smiles_all[i] for i in train_valid]
y_train = [y_all[i] for i in train_valid]
smiles_test = [smiles_all[i] for i in test_valid]
y_test = [y_all[i] for i in test_valid]

fit_local, val_local = train_test_split(np.arange(len(smiles_train)), test_size=0.20, random_state=SEED)
fit_local, val_local = list(fit_local), list(val_local)

mols_train = [molecules[i] for i in train_valid]
mols_fit = [mols_train[i] for i in fit_local]
mols_val = [mols_train[i] for i in val_local]
mols_test = [molecules[i] for i in test_valid]
y_fit = [y_train[i] for i in fit_local]
y_val = [y_train[i] for i in val_local]

print(f"Valid fit = {len(mols_fit)}; validation = {len(mols_val)}; final test = {len(mols_test)}")

metrics = []
preds = pd.DataFrame({"SMILES": smiles_test, "Y_TRUE_ESOL": y_test})

for desc_name, desc_factory in lazy.DESCRIPTORS.items():
    print("-" * 80)
    print(f"DESCRIPTOR: {desc_name}")
    try:
        calc = desc_factory()
        one_conf_bags = (
            [[mol] for mol in mols_fit]
            + [[mol] for mol in mols_val]
            + [[mol] for mol in mols_test]
        )

        X = lazy.calculate_descriptors(one_conf_bags, calc)
        nf, nv = len(mols_fit), len(mols_val)
        X_fit, X_val, X_test = X[:nf], X[nf:nf + nv], X[nf + nv:]
    except Exception as exc:
        print(f"Descriptor FAILED: {type(exc).__name__}: {exc}")
        for est_name in lazy.REGRESSORS:
            metrics.append({"descriptor": desc_name, "estimator": est_name, "model": f"{desc_name}|{est_name}", "N_test": len(y_test), "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Pearson_r": np.nan, "status": "DESCRIPTOR_FAILED"})
        continue

    for est_name, est_factory in lazy.REGRESSORS.items():
        model_name = f"{desc_name}|{est_name}"
        print(f"Running: {model_name}")
        try:
            estimator = est_factory(accelerator="cpu")
            _, _, pred = lazy.train_estimator(X_fit, X_val, X_test, y_fit, y_val, estimator, False, random_seed=SEED, accelerator="cpu")
            pred = np.asarray(pred, float)
            r2 = float(r2_score(y_test, pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            mae = float(mean_absolute_error(y_test, pred))
            r = pearson(y_test, pred)
            metrics.append({"descriptor": desc_name, "estimator": est_name, "model": model_name, "N_test": len(y_test), "R2": r2, "RMSE": rmse, "MAE": mae, "Pearson_r": r, "status": "OK"})
            preds[model_name] = pred
            print(f"  R2={r2:.6f} RMSE={rmse:.6f} MAE={mae:.6f} r={r:.6f}")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            metrics.append({"descriptor": desc_name, "estimator": est_name, "model": model_name, "N_test": len(y_test), "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Pearson_r": np.nan, "status": f"FAILED: {type(exc).__name__}: {exc}"})

pd.DataFrame(metrics).to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)
preds.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
pd.DataFrame({"SMILES": [smiles_all[i] for i in valid], "Y_TRUE_ESOL": [y_all[i] for i in valid], "RotB": [rotb[i] for i in valid], "selected_UFF_energy_kcal_mol": [energies[i] for i in valid], "maximum_UFF_energy_kcal_mol": [max_energies[i] for i in valid], "selected_conformer": [conf_ids[i] for i in valid]}).to_csv(os.path.join(OUTPUT_DIR, "selected_conformer_info.csv"), index=False)

print("=" * 80)
print("3D-SIL NEURAL BENCHMARK COMPLETE")
print(f"Successful models = {sum(x['status'] == 'OK' for x in metrics)}")
print(f"Failed models = {sum(x['status'] != 'OK' for x in metrics)}")
print("=" * 80)

