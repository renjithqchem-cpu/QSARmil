from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import f

ROOT = Path.home() / "Documents" / "qsarmil"

DATASETS = {
        "ESOL": ROOT / "data" / "esol_descriptor_architecture_interaction",
        "FreeSolv": ROOT / "data" / "freesolv_descriptor_architecture_interaction",
}

OUT = ROOT / "data" / "descriptor_architecture_formal_interaction"
OUT.mkdir(parents=True, exist_ok=True)


def parse_model(c):
        d, a = c.split("|", 1)
        return d, a


def load_errors(name, directory):
        if name == "ESOL":
                candidates = [
                        directory.parent / "esol_qsarmil_5conf" / "test.csv",
                        directory.parent / "esol_qsarmil_10conf" / "test.csv",
                        directory.parent / "esol_qsarmil_15conf" / "test.csv",
                        directory.parent / "esol_qsarmil_20conf" / "test.csv",
                ]

                testfile = next(
                        (p for p in candidates[::-1] if p.exists()),
                        None
                )

                if testfile is None:
                        raise FileNotFoundError(
                                "Could not locate ESOL QSARmil test.csv files."
                        )

                test = pd.read_csv(testfile)

                if "Y_TRUE" in test.columns:
                        target = pd.to_numeric(
                                test["Y_TRUE"],
                                errors="coerce"
                        ).to_numpy(float)

                else:
                        esol_file = ROOT / "data" / "delaney-processed.csv"
                        esol = pd.read_csv(esol_file)

                        if (
                                "Compound ID" in test.columns
                                and "Compound ID" in esol.columns
                        ):
                                lookup = dict(
                                        zip(
                                                esol["Compound ID"].astype(str),
                                                esol[
                                                        "measured log solubility in mols per litre"
                                                ].astype(float),
                                        )
                                )

                                target = (
                                        test["Compound ID"]
                                        .astype(str)
                                        .map(lookup)
                                        .to_numpy(float)
                                )

                        elif "SMILES" in test.columns:
                                lookup = dict(
                                        zip(
                                                esol["smiles"].astype(str),
                                                esol[
                                                        "measured log solubility in mols per litre"
                                                ].astype(float),
                                        )
                                )

                                target = (
                                        test["SMILES"]
                                        .astype(str)
                                        .map(lookup)
                                        .to_numpy(float)
                                )

                        else:
                                raise RuntimeError(
                                        "ESOL test.csv has neither Y_TRUE "
                                        "nor a usable Compound ID/SMILES column."
                                )

                        if np.any(~np.isfinite(target)):
                                raise RuntimeError(
                                        "Could not recover all ESOL test targets "
                                        "from delaney-processed.csv."
                                )

                model_cols = [
                        c for c in test.columns
                        if "|" in c
                ]

                if len(model_cols) != 72:
                        raise RuntimeError(
                                f"ESOL: expected 72 model columns, "
                                f"found {len(model_cols)}"
                        )

                nconf = testfile.parent.name

        else:
                testfile = (
                        ROOT
                        / "freesolv_qsarmil_5conf"
                        / "test.csv"
                )

                if not testfile.exists():
                        raise FileNotFoundError(
                                f"Missing {testfile}"
                        )

                test = pd.read_csv(testfile)

                bench = pd.read_csv(
                        ROOT / "data" / "freesolv_benchmark.csv"
                )

                if "Y_TRUE" in test.columns:
                        target = pd.to_numeric(
                                test["Y_TRUE"],
                                errors="coerce"
                        ).to_numpy(float)

                else:
                        lookup = dict(
                                zip(
                                        bench["smiles"].astype(str),
                                        bench[
                                                "experimental_dG_hyd_kcal_mol"
                                        ],
                                )
                        )

                        target = (
                                test["SMILES"]
                                .astype(str)
                                .map(lookup)
                                .to_numpy(float)
                        )

                        if np.any(~np.isfinite(target)):
                                raise RuntimeError(
                                        "Could not recover all FreeSolv "
                                        "test targets."
                                )

                model_cols = [
                        c for c in test.columns
                        if "|" in c
                ]

                if len(model_cols) != 72:
                        raise RuntimeError(
                                f"FreeSolv: expected 72 model columns, "
                                f"found {len(model_cols)}"
                        )

                nconf = "5conf"

        rows = []

        for i, c in enumerate(model_cols):
                d, a = parse_model(c)

                pred = pd.to_numeric(
                        test[c],
                        errors="coerce"
                ).to_numpy(float)

                ae = np.abs(pred - target)

                for j, val in enumerate(ae):
                        if np.isfinite(val):
                                rows.append(
                                        {
                                                "molecule": j,
                                                "descriptor": d,
                                                "architecture": a,
                                                "absolute_error": float(val),
                                        }
                                )

        return pd.DataFrame(rows), nconf


def fit_and_report(df, dataset, nconf):
        df = df.copy()

        df["molecule"] = df["molecule"].astype(str)
        df["descriptor"] = df["descriptor"].astype("category")
        df["architecture"] = df["architecture"].astype("category")

        reduced_formula = (
                "absolute_error ~ "
                "C(molecule) + "
                "C(descriptor) + "
                "C(architecture)"
        )

        full_formula = (
                "absolute_error ~ "
                "C(molecule) + "
                "C(descriptor) * C(architecture)"
        )

        reduced = smf.ols(
                reduced_formula,
                data=df
        ).fit()

        full = smf.ols(
                full_formula,
                data=df
        ).fit()

        ss_reduced = float(
                np.sum(reduced.resid ** 2)
        )

        ss_full = float(
                np.sum(full.resid ** 2)
        )

        df_reduced = int(
                reduced.df_resid
        )

        df_full = int(
                full.df_resid
        )

        df_num = df_reduced - df_full
        df_den = df_full

        ss_interaction = (
                ss_reduced - ss_full
        )

        F = (
                ss_interaction / df_num
        ) / (
                ss_full / df_den
        )

        p = float(
                f.sf(
                        F,
                        df_num,
                        df_den
                )
        )

        eta2_partial = (
                ss_interaction
                / (
                        ss_interaction
                        + ss_full
                )
        )

        desc_only = smf.ols(
                "absolute_error ~ "
                "C(molecule) + C(descriptor)",
                data=df
        ).fit()

        arch_only = smf.ols(
                "absolute_error ~ "
                "C(molecule) + C(architecture)",
                data=df
        ).fit()

        ss_desc = float(
                np.sum(desc_only.resid ** 2)
                - ss_reduced
        )

        ss_arch = float(
                np.sum(arch_only.resid ** 2)
                - ss_reduced
        )

        from statsmodels.stats.anova import anova_lm

        anova = anova_lm(
                full,
                typ=2
        )

        anova.to_csv(
                OUT
                / f"{dataset}_{nconf}_ANOVA_type2.csv"
        )

        return {
                "dataset": dataset,
                "conformer_count": nconf,
                "molecules": int(
                        df["molecule"].nunique()
                ),
                "observations": int(
                        len(df)
                ),
                "descriptor_levels": int(
                        df["descriptor"].nunique()
                ),
                "architecture_levels": int(
                        df["architecture"].nunique()
                ),
                "reduced_R2": float(
                        reduced.rsquared
                ),
                "full_R2": float(
                        full.rsquared
                ),
                "interaction_df": int(
                        df_num
                ),
                "residual_df_full": int(
                        df_den
                ),
                "interaction_F": float(
                        F
                ),
                "interaction_p": p,
                "interaction_partial_eta2": float(
                        eta2_partial
                ),
                "blocked_SS_total": float(
                        np.sum(
                                (
                                        df["absolute_error"]
                                        - df.groupby(
                                                "molecule"
                                        )[
                                                "absolute_error"
                                        ].transform("mean")
                                ) ** 2
                        )
                ),
                "descriptor_incremental_SS": ss_desc,
                "architecture_incremental_SS": ss_arch,
                "interaction_SS": ss_interaction,
        }


all_results = []

for dataset, directory in DATASETS.items():
        errors, nconf = load_errors(
                dataset,
                directory
        )

        errors.to_csv(
                OUT
                / f"{dataset}_{nconf}_molecule_absolute_errors.csv",
                index=False,
        )

        result = fit_and_report(
                errors,
                dataset,
                nconf
        )

        all_results.append(result)


results = pd.DataFrame(
        all_results
)

results.to_csv(
        OUT / "formal_interaction_test_results.csv",
        index=False
)

print()
print("=" * 80)
print("FORMAL DESCRIPTOR × ARCHITECTURE INTERACTION TEST")
print("=" * 80)
print()
print(
        "Analysis: molecule-blocked fixed-effects OLS"
)
print(
        "Response: per-molecule absolute prediction error"
)
print(
        "Reduced: molecule + descriptor + architecture"
)
print(
        "Full:    molecule + descriptor * architecture"
)
print()
print("RESULTS")
print("-" * 80)
print(
        results.to_string(
                index=False
        )
)

print()
print("INTERPRETATION")
print("-" * 80)

for _, r in results.iterrows():
        if r["interaction_p"] < 0.05:
                print(
                        f"{r['dataset']}: interaction is statistically "
                        f"detectable "
                        f"(F={r['interaction_F']:.3f}, "
                        f"p={r['interaction_p']:.3e}, "
                        f"partial eta^2="
                        f"{r['interaction_partial_eta2']:.4f})."
                )
        else:
                print(
                        f"{r['dataset']}: interaction is not "
                        f"statistically detectable "
                        f"(F={r['interaction_F']:.3f}, "
                        f"p={r['interaction_p']:.3e})."
                )

print()
print(
        f"Outputs: {OUT}"
)

