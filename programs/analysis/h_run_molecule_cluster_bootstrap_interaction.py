from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

ROOT = Path.home() / "Documents" / "qsarmil"
IN_DIR = ROOT / "data" / "descriptor_architecture_formal_interaction"
OUT_DIR = ROOT / "data" / "descriptor_architecture_bootstrap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
SEED = 20260921
rng = np.random.default_rng(SEED)


def interaction_stat(df):
        df = df.copy()
        df["molecule"] = df["molecule"].astype(str)
        df["descriptor"] = df["descriptor"].astype("category")
        df["architecture"] = df["architecture"].astype("category")

        reduced = smf.ols(
                "absolute_error ~ C(molecule) + C(descriptor) + C(architecture)",
                data=df
        ).fit()

        full = smf.ols(
                "absolute_error ~ C(molecule) + C(descriptor) * C(architecture)",
                data=df
        ).fit()

        ss_reduced = np.sum(reduced.resid ** 2)
        ss_full = np.sum(full.resid ** 2)

        df_num = reduced.df_resid - full.df_resid
        df_den = full.df_resid

        ss_int = ss_reduced - ss_full
        F = (ss_int / df_num) / (ss_full / df_den)
        eta2 = ss_int / (ss_int + ss_full)

        return float(F), float(eta2)


def load_dataset(dataset):
        path = IN_DIR / f"{dataset}_{'esol_qsarmil_20conf' if dataset == 'ESOL' else '5conf'}_molecule_absolute_errors.csv"

        if not path.exists():
                candidates = list(IN_DIR.glob(f"{dataset}_*_molecule_absolute_errors.csv"))
                if not candidates:
                        raise FileNotFoundError(
                                f"No molecule-level error file found for {dataset}"
                        )
                path = candidates[0]

        return pd.read_csv(path)


def bootstrap_dataset(df, dataset):
        molecule_ids = df["molecule"].unique()
        n_mol = len(molecule_ids)

        observed_F, observed_eta2 = interaction_stat(df)

        boot_F = np.empty(N_BOOT)
        boot_eta2 = np.empty(N_BOOT)

        for b in range(N_BOOT):
                sampled = rng.choice(
                        molecule_ids,
                        size=n_mol,
                        replace=True
                )

                parts = []

                for new_id, old_id in enumerate(sampled):
                        block = df[df["molecule"] == old_id].copy()
                        block["molecule"] = str(new_id)
                        parts.append(block)

                boot = pd.concat(parts, ignore_index=True)

                try:
                        boot_F[b], boot_eta2[b] = interaction_stat(boot)
                except Exception:
                        boot_F[b] = np.nan
                        boot_eta2[b] = np.nan

        valid_F = boot_F[np.isfinite(boot_F)]
        valid_eta2 = boot_eta2[np.isfinite(boot_eta2)]

        result = {
                "dataset": dataset,
                "n_molecules": n_mol,
                "n_bootstrap": N_BOOT,
                "valid_bootstrap": len(valid_F),
                "observed_F": observed_F,
                "observed_partial_eta2": observed_eta2,
                "F_mean": np.mean(valid_F),
                "F_median": np.median(valid_F),
                "F_2.5pct": np.percentile(valid_F, 2.5),
                "F_97.5pct": np.percentile(valid_F, 97.5),
                "eta2_mean": np.mean(valid_eta2),
                "eta2_median": np.median(valid_eta2),
                "eta2_2.5pct": np.percentile(valid_eta2, 2.5),
                "eta2_97.5pct": np.percentile(valid_eta2, 97.5),
                "bootstrap_fraction_F_gt_1": np.mean(valid_F > 1.0),
                "bootstrap_fraction_F_gt_observed": np.mean(
                        valid_F > observed_F
                ),
                "bootstrap_fraction_eta2_gt_0": np.mean(
                        valid_eta2 > 0
                ),
        }

        pd.DataFrame({
                "bootstrap_F": valid_F,
                "bootstrap_partial_eta2": valid_eta2,
        }).to_csv(
                OUT_DIR / f"{dataset}_bootstrap_distribution.csv",
                index=False
        )

        return result


all_results = []

for dataset in ["ESOL", "FreeSolv"]:
        print()
        print("=" * 80)
        print(f"MOLECULE-LEVEL CLUSTER BOOTSTRAP: {dataset}")
        print("=" * 80)

        df = load_dataset(dataset)

        print(f"Molecules: {df['molecule'].nunique()}")
        print(f"Observations: {len(df)}")
        print(f"Bootstrap replicates: {N_BOOT}")

        result = bootstrap_dataset(
                df,
                dataset
        )

        all_results.append(result)

        print()
        print(
                f"Observed F = {result['observed_F']:.6f}"
        )
        print(
                f"Observed partial eta^2 = "
                f"{result['observed_partial_eta2']:.6f}"
        )
        print(
                f"Bootstrap F 95% CI = "
                f"[{result['F_2.5pct']:.6f}, "
                f"{result['F_97.5pct']:.6f}]"
        )
        print(
                f"Bootstrap partial eta^2 95% CI = "
                f"[{result['eta2_2.5pct']:.6f}, "
                f"{result['eta2_97.5pct']:.6f}]"
        )

results = pd.DataFrame(all_results)

results.to_csv(
        OUT_DIR / "molecule_cluster_bootstrap_summary.csv",
        index=False
)

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(
        results.to_string(
                index=False
        )
)

print()
print(
        "Interpretation: this bootstrap resamples whole molecules, "
        "preserving all 72 descriptor × architecture errors within "
        "each sampled molecule. It provides an empirical stability "
        "interval for the interaction statistic; it is not treated "
        "as a conventional null-hypothesis p-value."
)

print()
print(f"Outputs: {OUT_DIR}")

