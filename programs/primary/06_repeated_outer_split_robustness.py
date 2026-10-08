from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, rdMolDescriptors

from qsarmil.modelling import lazy
from qsarmil.modelling.meta import MultiConformerRegressor


OUTER_SEEDS = [42, 123, 2024, 31415, 271828]
MIL_CONFORMERS = {"ESOL": 20, "FreeSolv": 5}

DESCRIPTOR_FUNCTIONS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Crippen.MolLogP),
    ("TPSA", rdMolDescriptors.CalcTPSA),
    ("HBD", Lipinski.NumHDonors),
    ("HBA", Lipinski.NumHAcceptors),
    ("RotB", Lipinski.NumRotatableBonds),
    ("RingCount", Lipinski.RingCount),
    ("AromaticRings", Lipinski.NumAromaticRings),
    ("HeavyAtomCount", lambda m: m.GetNumHeavyAtoms()),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
]


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Pearson_r": float(pearsonr(y_true, y_pred)[0]),
    }


def calculate_2d_descriptors(smiles):
    rows = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")

        rows.append([func(mol) for _, func in DESCRIPTOR_FUNCTIONS])

    return np.asarray(rows, dtype=float)


def lowest_uff_conformer(smiles, num_conf=20):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol_h = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1

    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            mol_h,
            numConfs=num_conf,
            params=params,
        )
    )

    if not conformer_ids:
        raise RuntimeError(f"Conformer generation failed: {smiles}")

    energies = []

    for cid in conformer_ids:
        try:
            AllChem.UFFOptimizeMolecule(
                mol_h,
                confId=cid,
                maxIters=2000,
            )
            force_field = AllChem.UFFGetMoleculeForceField(
                mol_h,
                confId=cid,
            )
            energy = float(force_field.CalcEnergy())

            if np.isfinite(energy):
                energies.append((cid, energy))

        except Exception:
            continue

    if not energies:
        raise RuntimeError(f"UFF optimization failed: {smiles}")

    best_cid = min(energies, key=lambda x: x[1])[0]

    selected = Chem.Mol(mol_h)
    selected.RemoveAllConformers()
    selected.AddConformer(
        mol_h.GetConformer(best_cid),
        assignId=True,
    )

    return Chem.RemoveHs(selected)


def load_dataset(case, root):
    root = Path(root)

    if case == "ESOL":
        data_file = root / "data" / "delaney-processed.csv"
        smiles_col = "smiles"
        target_col = "measured log solubility in mols per litre"

    elif case == "FreeSolv":
        data_file = root / "data" / "freesolv_benchmark.csv"
        smiles_col = "smiles"
        target_col = "experimental_dG_hyd_kcal_mol"

    else:
        raise ValueError(case)

    df = pd.read_csv(data_file)

    if smiles_col not in df.columns:
        raise KeyError(f"Missing SMILES column: {smiles_col}")

    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    return df, smiles_col, target_col


def make_outer_split(n_rows, seed):
    indices = np.arange(n_rows)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
    )

    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    if set(train_idx).intersection(set(test_idx)):
        raise RuntimeError("Outer train/test overlap detected")

    if len(set(train_idx).union(set(test_idx))) != n_rows:
        raise RuntimeError("Outer split does not cover all molecules")

    return train_idx, test_idx


def run_2d_rf(smiles_train, y_train, smiles_test, y_test):
    X_train = calculate_2d_descriptors(smiles_train)
    X_test = calculate_2d_descriptors(smiles_test)

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    return calculate_metrics(y_test, prediction)


def run_3d_sil(
    smiles_train,
    y_train,
    smiles_test,
    y_test,
    outer_seed,
):
    # The inner validation split is strictly inside the outer training set.
    local_indices = np.arange(len(smiles_train))

    fit_local, val_local = train_test_split(
        local_indices,
        test_size=0.20,
        random_state=outer_seed,
    )

    fit_local = list(fit_local)
    val_local = list(val_local)

    print("  Generating 20 conformers for 3D-SIL molecules...")

    all_smiles = list(smiles_train) + list(smiles_test)

    mols = []
    for i, smi in enumerate(all_smiles):
        mols.append(lowest_uff_conformer(smi, num_conf=20))

        if (i + 1) % 50 == 0 or i == len(all_smiles) - 1:
            print(f"    conformers: {i + 1}/{len(all_smiles)}")

    mol_fit = [mols[i] for i in fit_local]
    mol_val = [mols[i] for i in val_local]

    n_train = len(smiles_train)

    mol_test = [
        mols[n_train + i]
        for i in range(len(smiles_test))
    ]

    y_train = np.asarray(y_train, dtype=float)
    y_fit = y_train[fit_local]
    y_val = y_train[val_local]

    candidates = []
    test_predictions = {}

    for descriptor_name, descriptor_factory in lazy.DESCRIPTORS.items():

        print(f"  3D-SIL descriptor: {descriptor_name}")

        try:
            calculator = descriptor_factory()

            bags = [
                [mol]
                for mol in (mol_fit + mol_val + mol_test)
            ]

            X = lazy.calculate_descriptors(
                bags,
                calculator,
            )

            n_fit = len(mol_fit)
            n_val = len(mol_val)

            X_fit = X[:n_fit]
            X_val = X[n_fit:n_fit + n_val]
            X_test = X[n_fit + n_val:]

        except Exception as exc:
            print(
                f"    descriptor failed: {descriptor_name}: "
                f"{repr(exc)}"
            )
            continue

        for architecture_name, estimator_factory in lazy.REGRESSORS.items():

            model_name = (
                f"{descriptor_name}|{architecture_name}"
            )

            try:
                estimator = estimator_factory(
                    accelerator="cpu"
                )

                # HPO is disabled here deliberately. The explicit inner
                # validation set is the selection set for this outer split.
                _, validation_prediction, test_prediction = (
                    lazy.train_estimator(
                        X_fit,
                        X_val,
                        X_test,
                        y_fit,
                        y_val,
                        estimator,
                        False,
                        random_seed=outer_seed,
                        accelerator="cpu",
                    )
                )

                validation_prediction = np.asarray(
                    validation_prediction,
                    dtype=float,
                )

                test_prediction = np.asarray(
                    test_prediction,
                    dtype=float,
                )

                validation_rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            y_val,
                            validation_prediction,
                        )
                    )
                )

                validation_mae = float(
                    mean_absolute_error(
                        y_val,
                        validation_prediction,
                    )
                )

                candidates.append({
                    "model": model_name,
                    "descriptor": descriptor_name,
                    "architecture": architecture_name,
                    "validation_RMSE": validation_rmse,
                    "validation_MAE": validation_mae,
                })

                test_predictions[model_name] = test_prediction

            except Exception as exc:
                print(
                    f"    candidate failed: {model_name}: "
                    f"{repr(exc)}"
                )

    if not candidates:
        raise RuntimeError(
            "No valid 3D-SIL candidate models."
        )

    candidate_table = pd.DataFrame(candidates)

    candidate_table = candidate_table.sort_values(
        [
            "validation_RMSE",
            "validation_MAE",
            "model",
        ]
    ).reset_index(drop=True)

    selected = candidate_table.iloc[0]
    selected_model = selected["model"]

    prediction = test_predictions[selected_model]

    result = calculate_metrics(
        y_test,
        prediction,
    )

    selection = {
        "selected_model": selected_model,
        "selected_descriptor": selected["descriptor"],
        "selected_architecture": selected["architecture"],
        "validation_RMSE": float(selected["validation_RMSE"]),
        "validation_MAE": float(selected["validation_MAE"]),
    }

    return result, selection, candidate_table


def run_3d_mil(
    case,
    smiles_train,
    y_train,
    smiles_test,
    y_test,
    outer_seed,
    output_directory,
):
    n_conf = MIL_CONFORMERS[case]

    model_directory = (
        output_directory
        / f"{case}_seed{outer_seed}_MIL_{n_conf}conf"
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep the QSARmil production workflow unchanged.
    model = MultiConformerRegressor(
        num_conf=n_conf,
        hopt=False,
        verbose=True,
        random_seed=42,
        accelerator="cpu",
        output_folder=str(model_directory),
    )

    prediction = model.train_predict(
        list(smiles_train),
        np.asarray(y_train, dtype=float),
        list(smiles_test),
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    return calculate_metrics(
        y_test,
        prediction,
    )


def run_case(case, seed, root, output_root):
    df, smiles_col, target_col = load_dataset(
        case,
        root,
    )

    smiles = df[smiles_col].astype(str).to_numpy()
    y = df[target_col].astype(float).to_numpy()

    train_idx, test_idx = make_outer_split(
        len(df),
        seed,
    )

    smiles_train = smiles[train_idx]
    smiles_test = smiles[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    case_output = (
        Path(output_root)
        / case
        / f"seed_{seed}"
    )

    case_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print(f"{case} — OUTER SEED {seed}")
    print("=" * 80)
    print(
        f"Total={len(df)} "
        f"Train={len(train_idx)} "
        f"Test={len(test_idx)}"
    )

    # Save exact outer split before model fitting.
    split_table = pd.DataFrame({
        "row_index": np.concatenate(
            [train_idx, test_idx]
        ),
        "SMILES": np.concatenate(
            [smiles_train, smiles_test]
        ),
        "target": np.concatenate(
            [y_train, y_test]
        ),
        "split": (
            ["train"] * len(train_idx)
            + ["test"] * len(test_idx)
        ),
    })

    split_table.to_csv(
        case_output / "outer_split.csv",
        index=False,
    )

    all_results = []

    # --------------------------------------------------------
    # 2D-SIL
    # --------------------------------------------------------
    print()
    print("[1/3] 2D-SIL")

    start = time.time()

    metrics_2d = run_2d_rf(
        smiles_train,
        y_train,
        smiles_test,
        y_test,
    )

    elapsed = (time.time() - start) / 60.0

    all_results.append({
        "dataset": case,
        "outer_seed": seed,
        "model": "2D-SIL-RF",
        **metrics_2d,
        "runtime_min": elapsed,
    })

    print(
        "2D-SIL:",
        metrics_2d,
        f"runtime={elapsed:.2f} min",
    )

    # --------------------------------------------------------
    # 3D-SIL
    # --------------------------------------------------------
    print()
    print("[2/3] validation-selected 3D-SIL")

    start = time.time()

    metrics_3d_sil, selection, candidate_table = run_3d_sil(
        smiles_train,
        y_train,
        smiles_test,
        y_test,
        seed,
    )

    elapsed = (time.time() - start) / 60.0

    candidate_table.to_csv(
        case_output / "3d_sil_candidate_metrics.csv",
        index=False,
    )

    pd.DataFrame([selection]).to_csv(
        case_output / "3d_sil_selected_model.csv",
        index=False,
    )

    all_results.append({
        "dataset": case,
        "outer_seed": seed,
        "model": "3D-SIL",
        **metrics_3d_sil,
        "selected_model": selection["selected_model"],
        "selected_descriptor": selection["selected_descriptor"],
        "selected_architecture": selection["selected_architecture"],
        "validation_RMSE": selection["validation_RMSE"],
        "validation_MAE": selection["validation_MAE"],
        "runtime_min": elapsed,
    })

    print(
        "3D-SIL:",
        metrics_3d_sil,
        "selected=",
        selection["selected_model"],
        f"runtime={elapsed:.2f} min",
    )

    # --------------------------------------------------------
    # 3D-MIL
    # --------------------------------------------------------
    n_conf = MIL_CONFORMERS[case]

    print()
    print(
        f"[3/3] QSARmil 3D-MIL consensus "
        f"({n_conf} conformers)"
    )

    start = time.time()

    metrics_mil = run_3d_mil(
        case,
        smiles_train,
        y_train,
        smiles_test,
        y_test,
        seed,
        case_output,
    )

    elapsed = (time.time() - start) / 60.0

    all_results.append({
        "dataset": case,
        "outer_seed": seed,
        "model": f"3D-MIL-{n_conf}conf-consensus",
        **metrics_mil,
        "runtime_min": elapsed,
    })

    print(
        "3D-MIL:",
        metrics_mil,
        f"runtime={elapsed:.2f} min",
    )

    result_table = pd.DataFrame(all_results)

    result_table.to_csv(
        case_output / "metrics.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("SEED RESULT")
    print("=" * 80)
    print(
        result_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    return result_table


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        choices=["ESOL", "FreeSolv"],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=OUTER_SEEDS,
    )

    parser.add_argument(
        "--root",
        default=".",
    )

    parser.add_argument(
        "--out",
        default="data/repeated_outer_split_robustness",
    )

    args = parser.parse_args()

    result = run_case(
        args.case,
        args.seed,
        Path(args.root),
        Path(args.out),
    )

    print()
    print("Saved:")
    print(
        Path(args.out)
        / args.case
        / f"seed_{args.seed}"
    )


if __name__ == "__main__":
    main()

