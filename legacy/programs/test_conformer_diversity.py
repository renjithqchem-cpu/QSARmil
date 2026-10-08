import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from qsarmil.modelling.lazy import generate_conformers

INPUT = "data/esol_test.csv"
OUTPUT = "data/conformer_diversity_pilot.csv"

NCONF = 20
SEED = 42

df = pd.read_csv(INPUT).head(10)

smiles = df["smiles"].tolist()

print("=" * 80)
print("QSARmil CONFORMER DIVERSITY — 10 MOLECULE PILOT")
print("=" * 80)

confs_all = generate_conformers(
    smiles,
    num_conf=NCONF,
    num_cpu=1,
    verbose=True,
    random_seed=SEED,
)

rows = []

for idx, (smi, confs) in enumerate(zip(smiles, confs_all)):

    print(f"\nMolecule {idx+1}/10")

    if not isinstance(confs, list) or len(confs) == 0:
        print("  FAILED")
        continue

    n = len(confs)

    # --------------------------------------------------------
    # UFF energies
    # --------------------------------------------------------
    energies = []

    for mol in confs:
        conf_id = mol.GetConformer().GetId()

        ff = AllChem.UFFGetMoleculeForceField(
            mol,
            confId=conf_id
        )

        energies.append(ff.CalcEnergy())

    energies = np.asarray(energies, dtype=float)

    e_min = energies.min()
    e_max = energies.max()

    rel_e = energies - e_min

    # --------------------------------------------------------
    # Pairwise RMSD
    # --------------------------------------------------------
    rmsds = []

    for i in range(n):
        for j in range(i + 1, n):

            rms = AllChem.GetBestRMS(
                confs[i],
                confs[j]
            )

            rmsds.append(rms)

    rmsds = np.asarray(rmsds, dtype=float)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------
    rows.append({
        "row": idx,
        "smiles": smi,
        "n_generated": n,
        "uff_E_min_kcal_mol": e_min,
        "uff_E_max_kcal_mol": e_max,
        "uff_delta_E_kcal_mol": e_max - e_min,
        "uff_mean_relative_E_kcal_mol": rel_e.mean(),
        "uff_median_relative_E_kcal_mol": np.median(rel_e),
        "rmsd_mean_A": rmsds.mean(),
        "rmsd_median_A": np.median(rmsds),
        "rmsd_max_A": rmsds.max(),
    })

    print(f"  conformers      : {n}")
    print(f"  UFF ΔE          : {e_max-e_min:.4f} kcal/mol")
    print(f"  mean RMSD       : {rmsds.mean():.4f} Å")
    print(f"  max RMSD        : {rmsds.max():.4f} Å")

out = pd.DataFrame(rows)
out.to_csv(OUTPUT, index=False)

print("\n" + "=" * 80)
print("PILOT COMPLETE")
print("=" * 80)
print(out.to_string(index=False))
print(f"\nSaved: {OUTPUT}")
