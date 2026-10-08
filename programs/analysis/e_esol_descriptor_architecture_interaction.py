import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

BASE = os.path.expanduser("~/Documents/qsarmil")
INPUT = os.path.join(BASE, "data", "esol_72model_audit")
OUT = os.path.join(
    BASE, "data", "esol_descriptor_architecture_interaction"
)
os.makedirs(OUT, exist_ok=True)

CONF_COUNTS = [5, 10, 15, 20]

DESCRIPTORS = [
    "RDKitGEOM",
    "RDKitAUTOCORR",
    "RDKitRDF",
    "RDKitMORSE",
    "RDKitWHIM",
    "MolFeatUSRD",
    "MolFeatElectroShape",
    "RDKitGETAWAY",
    "MolFeatPmapper",
]

ARCHITECTURES = [
    "MeanInstanceWrapperMLPNetworkRegressor",
    "MeanBagWrapperMLPNetworkRegressor",
    "MeanBagNetworkRegressor",
    "MeanInstanceNetworkRegressor",
    "AdditiveAttentionNetworkRegressor",
    "SelfAttentionNetworkRegressor",
    "HopfieldAttentionNetworkRegressor",
    "DynamicPoolingNetworkRegressor",
]


def parse_model_name(name):
    name = str(name)

    for descriptor in DESCRIPTORS:
        prefix = descriptor + "|"

        if name.startswith(prefix):
            architecture = name[len(prefix):]

            if architecture in ARCHITECTURES:
                return descriptor, architecture

    return None, None


def load_prediction_file(nconf):

    path = os.path.join(
        INPUT,
        f"matched_test_predictions_{nconf}conf.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    print("\n" + "=" * 75)
    print(f"{nconf}-CONFORMER DATASET")
    print("=" * 75)

    print("Shape:", df.shape)

    return df


def identify_target_column(df):

    if "Y_TRUE_ESOL" not in df.columns:
        raise RuntimeError(
            "Y_TRUE_ESOL was not found."
        )

    return "Y_TRUE_ESOL"


def get_model_columns(df):

    model_cols = []

    for column in df.columns:

        descriptor, architecture = parse_model_name(column)

        if descriptor is not None:
            model_cols.append(column)

    expected = len(DESCRIPTORS) * len(ARCHITECTURES)

    if len(model_cols) != expected:

        print(
            f"\nERROR: expected {expected} models, "
            f"found {len(model_cols)}."
        )

        missing = []

        for d in DESCRIPTORS:
            for a in ARCHITECTURES:

                name = f"{d}|{a}"

                if name not in model_cols:
                    missing.append(name)

        print("\nMissing models:")

        for name in missing:
            print(" ", name)

        raise RuntimeError(
            "Model-column validation failed."
        )

    return model_cols


def calculate_metrics(y, pred):

    error = pred - y

    rmse = np.sqrt(np.mean(error ** 2))

    mae = np.mean(np.abs(error))

    ss_res = np.sum(error ** 2)

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    r2 = 1.0 - ss_res / ss_tot

    pearson = np.corrcoef(
        y,
        pred
    )[0, 1]

    spearman = spearmanr(
        y,
        pred
    ).statistic

    return (
        rmse,
        mae,
        r2,
        pearson,
        spearman
    )


# ================================================================
# READ ALL PREDICTIONS
# ================================================================

results = []

error_arrays = {}

smiles_reference = None
target_reference = None

for nconf in CONF_COUNTS:

    df = load_prediction_file(nconf)

    target_col = identify_target_column(df)

    model_cols = get_model_columns(df)

    print("Target column:", target_col)
    print("Validated model columns:", len(model_cols))

    y = pd.to_numeric(
        df[target_col],
        errors="coerce"
    ).to_numpy()

    if np.any(~np.isfinite(y)):
        raise RuntimeError(
            f"Non-finite target values in {nconf}-conformer file."
        )

    if smiles_reference is None:
        smiles_reference = df["SMILES"].astype(str).tolist()
        target_reference = y.copy()

    else:

        if df["SMILES"].astype(str).tolist() != smiles_reference:
            raise RuntimeError(
                f"SMILES ordering differs for {nconf} conformers."
            )

        if not np.allclose(
            y,
            target_reference,
            rtol=0,
            atol=1e-12
        ):
            raise RuntimeError(
                f"Y_TRUE_ESOL differs for {nconf} conformers."
            )

    for column in model_cols:

        descriptor, architecture = parse_model_name(column)

        pred = pd.to_numeric(
            df[column],
            errors="coerce"
        ).to_numpy()

        if np.any(~np.isfinite(pred)):
            raise RuntimeError(
                f"Non-finite predictions in {column}."
            )

        rmse, mae, r2, pearson, spearman = calculate_metrics(
            y,
            pred
        )

        absolute_error = np.abs(
            pred - y
        )

        error_arrays[
            (nconf, descriptor, architecture)
        ] = absolute_error

        results.append({
            "n_conformers": nconf,
            "descriptor": descriptor,
            "architecture": architecture,
            "model": column,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Pearson": pearson,
            "Spearman": spearman,
        })


results = pd.DataFrame(results)

print("\nTotal model records:", len(results))

expected_records = (
    len(CONF_COUNTS)
    * len(DESCRIPTORS)
    * len(ARCHITECTURES)
)

if len(results) != expected_records:
    raise RuntimeError(
        f"Expected {expected_records} records, "
        f"found {len(results)}."
    )


results.to_csv(
    os.path.join(
        OUT,
        "all_72_model_metrics.csv"
    ),
    index=False
)


# ================================================================
# PERFORMANCE MATRICES
# ================================================================

for nconf in CONF_COUNTS:

    subset = results[
        results["n_conformers"] == nconf
    ]

    rmse = subset.pivot(
        index="descriptor",
        columns="architecture",
        values="RMSE"
    )

    mae = subset.pivot(
        index="descriptor",
        columns="architecture",
        values="MAE"
    )

    r2 = subset.pivot(
        index="descriptor",
        columns="architecture",
        values="R2"
    )

    rmse.to_csv(
        os.path.join(
            OUT,
            f"RMSE_matrix_{nconf}conf.csv"
        )
    )

    mae.to_csv(
        os.path.join(
            OUT,
            f"MAE_matrix_{nconf}conf.csv"
        )
    )

    r2.to_csv(
        os.path.join(
            OUT,
            f"R2_matrix_{nconf}conf.csv"
        )
    )


# ================================================================
# DESCRIPTOR MAIN EFFECT
# ================================================================

descriptor_rows = []

for nconf in CONF_COUNTS:

    subset = results[
        results["n_conformers"] == nconf
    ]

    for descriptor in DESCRIPTORS:

        values = subset.loc[
            subset["descriptor"] == descriptor,
            "RMSE"
        ].to_numpy()

        descriptor_rows.append({
            "n_conformers": nconf,
            "descriptor": descriptor,
            "mean_RMSE": np.mean(values),
            "median_RMSE": np.median(values),
            "std_RMSE": np.std(values, ddof=1),
            "min_RMSE": np.min(values),
            "max_RMSE": np.max(values),
        })


descriptor_effects = pd.DataFrame(
    descriptor_rows
)

descriptor_effects.to_csv(
    os.path.join(
        OUT,
        "descriptor_main_effects.csv"
    ),
    index=False
)


# ================================================================
# ARCHITECTURE MAIN EFFECT
# ================================================================

architecture_rows = []

for nconf in CONF_COUNTS:

    subset = results[
        results["n_conformers"] == nconf
    ]

    for architecture in ARCHITECTURES:

        values = subset.loc[
            subset["architecture"] == architecture,
            "RMSE"
        ].to_numpy()

        architecture_rows.append({
            "n_conformers": nconf,
            "architecture": architecture,
            "mean_RMSE": np.mean(values),
            "median_RMSE": np.median(values),
            "std_RMSE": np.std(values, ddof=1),
            "min_RMSE": np.min(values),
            "max_RMSE": np.max(values),
        })


architecture_effects = pd.DataFrame(
    architecture_rows
)

architecture_effects.to_csv(
    os.path.join(
        OUT,
        "architecture_main_effects.csv"
    ),
    index=False
)


# ================================================================
# DESCRIPTOR × ARCHITECTURE INTERACTION
# ================================================================

interaction_rows = []

for nconf in CONF_COUNTS:

    subset = results[
        results["n_conformers"] == nconf
    ].copy()

    grand_mean = subset["RMSE"].mean()

    descriptor_means = (
        subset
        .groupby("descriptor")["RMSE"]
        .mean()
        .to_dict()
    )

    architecture_means = (
        subset
        .groupby("architecture")["RMSE"]
        .mean()
        .to_dict()
    )

    ss_total = np.sum(
        (subset["RMSE"] - grand_mean) ** 2
    )

    ss_descriptor = 0.0

    for descriptor in DESCRIPTORS:

        ss_descriptor += 8 * (
            descriptor_means[descriptor]
            - grand_mean
        ) ** 2

    ss_architecture = 0.0

    for architecture in ARCHITECTURES:

        ss_architecture += 9 * (
            architecture_means[architecture]
            - grand_mean
        ) ** 2

    ss_interaction = 0.0

    for _, row in subset.iterrows():

        additive_prediction = (
            grand_mean
            + descriptor_means[row["descriptor"]]
            - grand_mean
            + architecture_means[row["architecture"]]
            - grand_mean
        )

        interaction_residual = (
            row["RMSE"]
            - additive_prediction
        )

        ss_interaction += (
            interaction_residual ** 2
        )

    interaction_rows.append({
        "n_conformers": nconf,
        "SS_total": ss_total,
        "SS_descriptor": ss_descriptor,
        "SS_architecture": ss_architecture,
        "SS_interaction": ss_interaction,
        "descriptor_fraction": (
            ss_descriptor / ss_total
        ),
        "architecture_fraction": (
            ss_architecture / ss_total
        ),
        "interaction_fraction": (
            ss_interaction / ss_total
        ),
    })


interaction = pd.DataFrame(
    interaction_rows
)

interaction.to_csv(
    os.path.join(
        OUT,
        "descriptor_architecture_variance_decomposition.csv"
    ),
    index=False
)


# ================================================================
# PAIRED DESCRIPTOR COMPARISONS
# ================================================================

descriptor_pairs = []

for nconf in CONF_COUNTS:

    for architecture in ARCHITECTURES:

        for i, d1 in enumerate(DESCRIPTORS):

            e1 = error_arrays[
                (nconf, d1, architecture)
            ]

            for d2 in DESCRIPTORS[i + 1:]:

                e2 = error_arrays[
                    (nconf, d2, architecture)
                ]

                difference = e1 - e2

                try:

                    test = wilcoxon(
                        difference,
                        zero_method="wilcox",
                        alternative="two-sided"
                    )

                    p = test.pvalue

                except ValueError:

                    p = np.nan

                descriptor_pairs.append({
                    "n_conformers": nconf,
                    "architecture": architecture,
                    "descriptor_1": d1,
                    "descriptor_2": d2,
                    "mean_difference": np.mean(difference),
                    "median_difference": np.median(difference),
                    "fraction_d1_better": np.mean(
                        e1 < e2
                    ),
                    "wilcoxon_p": p,
                })


descriptor_pairs = pd.DataFrame(
    descriptor_pairs
)

descriptor_pairs.to_csv(
    os.path.join(
        OUT,
        "paired_descriptor_comparisons.csv"
    ),
    index=False
)


# ================================================================
# PAIRED ARCHITECTURE COMPARISONS
# ================================================================

architecture_pairs = []

for nconf in CONF_COUNTS:

    for descriptor in DESCRIPTORS:

        for i, a1 in enumerate(ARCHITECTURES):

            e1 = error_arrays[
                (nconf, descriptor, a1)
            ]

            for a2 in ARCHITECTURES[i + 1:]:

                e2 = error_arrays[
                    (nconf, descriptor, a2)
                ]

                difference = e1 - e2

                try:

                    test = wilcoxon(
                        difference,
                        zero_method="wilcox",
                        alternative="two-sided"
                    )

                    p = test.pvalue

                except ValueError:

                    p = np.nan

                architecture_pairs.append({
                    "n_conformers": nconf,
                    "descriptor": descriptor,
                    "architecture_1": a1,
                    "architecture_2": a2,
                    "mean_difference": np.mean(difference),
                    "median_difference": np.median(difference),
                    "fraction_a1_better": np.mean(
                        e1 < e2
                    ),
                    "wilcoxon_p": p,
                })


architecture_pairs = pd.DataFrame(
    architecture_pairs
)

architecture_pairs.to_csv(
    os.path.join(
        OUT,
        "paired_architecture_comparisons.csv"
    ),
    index=False
)


# ================================================================
# MODEL TRAJECTORIES ACROSS CONFORMER COUNTS
# ================================================================

trajectory_rows = []

for descriptor in DESCRIPTORS:

    for architecture in ARCHITECTURES:

        values = []

        for nconf in CONF_COUNTS:

            row = results[
                (results["n_conformers"] == nconf)
                &
                (results["descriptor"] == descriptor)
                &
                (results["architecture"] == architecture)
            ]

            values.append(
                row["RMSE"].iloc[0]
            )

        rho, p = spearmanr(
            CONF_COUNTS,
            values
        )

        trajectory_rows.append({
            "descriptor": descriptor,
            "architecture": architecture,
            "RMSE_5conf": values[0],
            "RMSE_10conf": values[1],
            "RMSE_15conf": values[2],
            "RMSE_20conf": values[3],
            "Spearman_conformer_count_RMSE": rho,
            "p_value": p,
        })


trajectories = pd.DataFrame(
    trajectory_rows
)

trajectories.to_csv(
    os.path.join(
        OUT,
        "model_conformer_count_trajectories.csv"
    ),
    index=False
)


# ================================================================
# TERMINAL SUMMARY
# ================================================================

print("\n")
print("=" * 75)
print("DESCRIPTOR × ARCHITECTURE INTERACTION ANALYSIS")
print("=" * 75)

for nconf in CONF_COUNTS:

    subset = results[
        results["n_conformers"] == nconf
    ]

    best = subset.loc[
        subset["RMSE"].idxmin()
    ]

    worst = subset.loc[
        subset["RMSE"].idxmax()
    ]

    dec = interaction[
        interaction["n_conformers"] == nconf
    ].iloc[0]

    print(f"\n{nconf} conformers")
    print("-" * 75)

    print(
        f"Best model : "
        f"{best['descriptor']} | "
        f"{best['architecture']} | "
        f"RMSE={best['RMSE']:.6f}"
    )

    print(
        f"Worst model: "
        f"{worst['descriptor']} | "
        f"{worst['architecture']} | "
        f"RMSE={worst['RMSE']:.6f}"
    )

    print(
        f"Descriptor variance fraction : "
        f"{dec['descriptor_fraction']:.4f}"
    )

    print(
        f"Architecture variance fraction: "
        f"{dec['architecture_fraction']:.4f}"
    )

    print(
        f"Interaction variance fraction : "
        f"{dec['interaction_fraction']:.4f}"
    )


print("\n")
print("=" * 75)
print("OUTPUT DIRECTORY")
print("=" * 75)

print(OUT)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUT)):
    print(" ", filename)

print("\nAnalysis completed successfully.")
