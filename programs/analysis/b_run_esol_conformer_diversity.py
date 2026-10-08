import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from qsarmil.modelling.lazy import generate_conformers


INPUT = "data/esol_test.csv"
OUTPUT = "data/esol_conformer_diversity_20conf.csv"

NCONF = 20
SEED = 42
NUM_CPU = 16
LOW_E_CUTOFF = 5.0


df = pd.read_csv(INPUT)

print("=" * 80)
print("ESOL 20-CONFORMER DIVERSITY ANALYSIS")
print("=" * 80)
print(f"Molecules          : {len(df)}")
print(f"Conformers         : {NCONF}")
print(f"Random seed        : {SEED}")
print(f"Low-energy cutoff  : {LOW_E_CUTOFF} kcal/mol")
print()


smiles = df["smiles"].tolist()

confs_all = generate_conformers(
    smiles,
    num_conf=NCONF,
    num_cpu=NUM_CPU,
    verbose=True,
    random_seed=SEED,
)


rows = []

for idx, (smi, confs) in enumerate(zip(smiles, confs_all)):

    print(f"[{idx + 1}/{len(smiles)}] {smi}")

    result = {
        "row": idx,
        "smiles": smi,
        "n_requested": NCONF,
        "n_generated": 0,
        "uff_E_min_kcal_mol": np.nan,
        "uff_E_max_kcal_mol": np.nan,
        "uff_delta_E_kcal_mol": np.nan,
        "uff_mean_relative_E_kcal_mol": np.nan,
        "uff_median_relative_E_kcal_mol": np.nan,
        "n_low_energy_5kcal": 0,
        "rmsd_mean_A": np.nan,
        "rmsd_median_A": np.nan,
        "rmsd_max_A": np.nan,
        "rmsd_mean_lowE5_A": np.nan,
        "rmsd_median_lowE5_A": np.nan,
        "rmsd_max_lowE5_A": np.nan,
        "status": "OK",
    }

    if not isinstance(confs, list) or len(confs) == 0:
        result["status"] = "CONFORMER_GENERATION_FAILED"
        rows.append(result)
        print("  FAILED: no conformers")
        continue

    n = len(confs)
    result["n_generated"] = n

    # --------------------------------------------------------
    # UFF energies
    # --------------------------------------------------------

    energies = []

    try:
        for mol in confs:

            conf_id = mol.GetConformer().GetId()

            ff = AllChem.UFFGetMoleculeForceField(
                mol,
                confId=conf_id
            )

            energies.append(ff.CalcEnergy())

    except Exception as exc:

        result["status"] = "UFF_FAILED"
        rows.append(result)

        print(f"  UFF FAILED: {exc}")
        continue

    energies = np.asarray(energies, dtype=float)

    e_min = float(np.min(energies))
    e_max = float(np.max(energies))

    relative_energy = energies - e_min

    low_mask = relative_energy <= LOW_E_CUTOFF
    n_low = int(np.sum(low_mask))

    result["uff_E_min_kcal_mol"] = e_min
    result["uff_E_max_kcal_mol"] = e_max
    result["uff_delta_E_kcal_mol"] = e_max - e_min
    result["uff_mean_relative_E_kcal_mol"] = float(
        np.mean(relative_energy)
    )
    result["uff_median_relative_E_kcal_mol"] = float(
        np.median(relative_energy)
    )
    result["n_low_energy_5kcal"] = n_low

    # --------------------------------------------------------
    # Pairwise RMSD — all conformers
    # --------------------------------------------------------

    rmsds = []

    for i in range(n):
        for j in range(i + 1, n):

            try:
                rms = AllChem.GetBestRMS(
                    confs[i],
                    confs[j]
                )
                rmsds.append(float(rms))

            except Exception:
                pass

    if rmsds:

        rmsds = np.asarray(rmsds, dtype=float)

        result["rmsd_mean_A"] = float(np.mean(rmsds))
        result["rmsd_median_A"] = float(np.median(rmsds))
        result["rmsd_max_A"] = float(np.max(rmsds))

    # --------------------------------------------------------
    # Pairwise RMSD — low-energy conformers
    # --------------------------------------------------------

    low_indices = np.where(low_mask)[0]

    low_rmsds = []

    if len(low_indices) >= 2:

        for a in range(len(low_indices)):
            for b in range(a + 1, len(low_indices)):

                i = int(low_indices[a])
                j = int(low_indices[b])

                try:
                    rms = AllChem.GetBestRMS(
                        confs[i],
                        confs[j]
                    )
                    low_rmsds.append(float(rms))

                except Exception:
                    pass

    if low_rmsds:

        low_rmsds = np.asarray(low_rmsds, dtype=float)

        result["rmsd_mean_lowE5_A"] = float(
            np.mean(low_rmsds)
        )
        result["rmsd_median_lowE5_A"] = float(
            np.median(low_rmsds)
        )
        result["rmsd_max_lowE5_A"] = float(
            np.max(low_rmsds)
        )

    print(
        f"  n={n} | "
        f"UFF dE={result['uff_delta_E_kcal_mol']:.3f} | "
        f"RMSD mean={result['rmsd_mean_A']:.3f} | "
        f"RMSD max={result['rmsd_max_A']:.3f}"
    )

    rows.append(result)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out = pd.DataFrame(rows)

out.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)

print(f"Total molecules : {len(out)}")
print(
    f"Successful       : {(out['status'] == 'OK').sum()}"
)
print(
    f"Failed           : {(out['status'] != 'OK').sum()}"
)

print()
print(out["n_generated"].value_counts().sort_index())

print()
print(f"Saved: {OUTPUT}")
