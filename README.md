# QSARmil: Conformer-Ensemble Property Prediction Reproducibility Package

This repository contains the production programs, analysis programs, benchmark inputs, saved model outputs, statistical analysis outputs, diagnostics, and selected historical programs associated with:

**Representation, Aggregation, and Model Selection in Conformer-Ensemble Property Prediction: ESOL and FreeSolv Benchmarks**

The repository is organized to make the computational workflow transparent and reproducible while preserving the program names used during the study.

---

## Repository structure

```text
QSARmil_GitHub_reproducibility/
│
├── README.md
├── requirements.in
├── requirements-lock.txt
├── ENVIRONMENT.md
│
├── programs/
│   ├── primary/
│   └── analysis/
│
├── data/
│   ├── input/
│   ├── model_runs/
│   ├── primary_outputs/
│   ├── analysis_outputs/
│   └── diagnostics/
│
├── diagnostics/
│   └── programs/
│
├── legacy/
│   └── programs/

```

### `programs/primary/`

These are the principal production programs used for the benchmark workflows. The numerical prefixes indicate the recommended reading/reproduction order; they do not imply that every program is a strict execution dependency of the next.

```text
01_prepare_esol.py
02_run_esol_sil.py
03_run_esol_qsarmil_3conf.py
04_validation_selected_3d_sil.py
05_run_freesolv_2d_sil.py
06_repeated_outer_split_robustness.py
```

The primary workflow covers:

1. preparation of the fixed ESOL benchmark split;
2. the 2D-SIL ESOL baseline;
3. the QSARmil 3D-MIL workflow;
4. validation-selected 3D-SIL;
5. the FreeSolv 2D-SIL baseline; and
6. repeated outer-split robustness analysis.

The benchmark comparison is a comparison of complete representation-to-model workflows. The 2D baseline uses a Random Forest estimator, whereas QSARmil uses neural estimators; therefore the repository does not treat the comparison as an isolated learner-controlled experiment.

### `programs/analysis/`

These programs reproduce manuscript and Supporting Information analyses. The letter prefixes indicate the recommended scientific reading order while preserving the original descriptive filenames.

```text
a_run_esol_analysis.py
b_run_esol_conformer_diversity.py
c_esol_288_model_statistical_analysis.py
d_esol_conformer_effect_statistics.py
e_esol_descriptor_architecture_interaction.py
f_esol_descriptor_architecture_fdr.py
g_run_formal_descriptor_architecture_interaction.py
h_run_molecule_cluster_bootstrap_interaction.py
i_run_freesolv_5conf_descriptor_architecture_analysis.py
j_run_freesolv_5conf_descriptor_architecture_fdr.py
k_compare_3d_sil_hopt_vs_3d_mil.py
l_analyze_3d_sil_hopt_vs_3d_mil_stats.py
m_analyze_molecule_level_3d_sil_hopt_vs_3d_mil.py
n_analyze_molecule_improvement_matrix.py
o_analyze_model_conformer_trajectories.py
p_esol_molecule_level_mil_vs_sil.py
q_analyze_freesolv_descriptor_domain.py
r_freesolv_domain_quartile_analysis.py
```

These analyses include conformer-count trajectories, descriptor/architecture effects, multiple-comparison analyses, interaction tests, molecule-level comparisons, model/conformer trajectories, and FreeSolv domain analyses.

### `diagnostics/programs/`

This directory contains diagnostic, stability, audit, and forensic programs used to examine workflow behavior and numerical stability. These programs are intentionally separated from the primary and manuscript-analysis workflows.

They include audits of model stability, descriptor/conformer integration, QSARmil model outputs, validation stability, formal interaction labeling, bootstrap stability, FreeSolv behavior, and the seed-123 FreeSolv robustness case.

### `legacy/programs/`

This directory contains superseded or historical programs retained for provenance. They are not part of the recommended production workflow.

---

## Benchmark datasets

The repository contains the benchmark inputs and frozen splits used by the study under `data/input/`.

### ESOL

The repository contains the frozen ESOL benchmark data used for the study, including the fixed seed-42 molecule-level split.

### FreeSolv

The repository contains the FreeSolv benchmark data and the fixed seed-42 train/test split used for the primary benchmark.

The repository preserves the study inputs and derived benchmark files used in the reported analyses. Dataset files are not represented as newly generated experimental measurements.

---

## Primary model design

The study evaluates three main representation-to-model workflows:

- **2D-SIL**: a 2D molecular-descriptor baseline using the selected Random Forest workflow.
- **3D-SIL**: a single-conformer workflow in which the lowest-UFF-energy conformer is selected without using the target property, followed by validation-based descriptor/architecture selection.
- **3D-MIL**: a multi-conformer QSARmil workflow in which molecular conformers are represented as an ensemble and candidate descriptor/architecture combinations are evaluated.

The conformers were generated algorithmically using ETKDGv3/UFF. UFF energies were used for conformer selection where specified; they were not interpreted as equilibrium populations, and no Boltzmann weighting was applied.

The 3D-MIL benchmark explores descriptor identity, aggregation/model architecture, conformer multiplicity, and model selection together. It is therefore not a causal conformer-count-only experiment.

---

## Descriptor families and model architectures

The QSARmil candidate landscape includes the descriptor families evaluated in the manuscript:

- RDKitGEOM
- RDKitAUTOCORR
- RDKitRDF
- RDKitMORSE
- RDKitWHIM
- MolFeatUSRD
- MolFeatElectroShape
- RDKitGETAWAY
- MolFeatPmapper

The evaluated architecture set includes:

- MeanInstanceWrapperMLPNetworkRegressor
- MeanBagWrapperMLPNetworkRegressor
- MeanBagNetworkRegressor
- MeanInstanceNetworkRegressor
- AdditiveAttentionNetworkRegressor
- SelfAttentionNetworkRegressor
- HopfieldAttentionNetworkRegressor
- DynamicPoolingNetworkRegressor

---

## Fixed benchmark split

The primary fixed split is a molecule-level **80:20 split with random seed 42**.

The resulting benchmark sizes are:

| Dataset | Training | Test |
|---|---:|---:|
| ESOL | 902 | 226 |
| FreeSolv | 513 | 129 |

The fixed split is preserved in the repository so that the reported primary benchmark can be traced to a defined molecule-level partition.

---

## Conformer generation

The conformer workflows use:

- ETKDGv3 conformer generation
- UFF optimization
- random seed 42 for the fixed production realization
- no Boltzmann weighting

For ESOL, conformer-count analyses include 3, 5, 10, 15, and 20 conformers.

The study treats these as **generated conformer collections**, not experimentally observed equilibrium ensembles.

---

## Model selection

For the validation-selected 3D-SIL workflow, candidate descriptor/architecture combinations are evaluated on the inner validation set. The selected candidate is then refit using the full outer training set and evaluated once on the held-out test set.

For the 3D-MIL workflow, the candidate landscape contains:

```text
9 descriptor families × 8 architectures = 72 candidate combinations
```

The reported consensus results are based on the production workflow and saved outputs included in the repository.

---

## Data and output organization

### `data/input/`

Frozen benchmark inputs and split files.

### `data/model_runs/`

Saved train/validation/test outputs from the principal QSARmil model runs.

### `data/primary_outputs/`

Primary numerical outputs used by the manuscript, including:

- model metrics;
- predictions;
- conformer-scaling results;
- descriptor and architecture effects;
- variance decomposition;
- interaction results;
- molecule-level improvement summaries;
- domain analyses; and
- repeated-split metrics.

### `data/analysis_outputs/`

Derived tables used by the manuscript/SI statistical analyses, including paired descriptor and architecture comparisons, FDR-adjusted results, model trajectories, flexibility/improvement analyses, and FreeSolv domain-error analyses.

### `data/diagnostics/`

Saved outputs associated with diagnostic and forensic analyses.

---

## Reproducibility environment

The computational environment used for the study was Python 3.11.16.

Core package versions are pinned in `requirements.in`:

```text
numpy==1.26.4
pandas
scipy==1.17.1
scikit-learn==1.9.1
statsmodels
matplotlib==3.10.9
rdkit==2026.03.6
molfeat==1.0.0
qsarmil==1.0.8
qsarcons==1.1.6
mikit-learn==1.0.6
```

The exact environment snapshot is preserved separately in `requirements-lock.txt`. That lock file records the package environment used for the computational study, including packages whose exact builds originated from the study's conda-based environment.

`qsarmil` is used as an external package dependency; the repository contains the study-specific production and analysis programs rather than the upstream `qsarmil` package source.

---

## Running the primary workflows

From the repository root:

```bash
python programs/primary/01_prepare_esol.py
python programs/primary/02_run_esol_sil.py
python programs/primary/03_run_esol_qsarmil_3conf.py
python programs/primary/04_validation_selected_3d_sil.py
python programs/primary/05_run_freesolv_2d_sil.py
python programs/primary/06_repeated_outer_split_robustness.py
```

The scripts are organized by benchmark component. Some workflows are independent rather than strict sequential dependencies.

Analysis programs are run separately after the relevant primary outputs are available. For example:

```bash
python programs/analysis/a_run_esol_analysis.py
python programs/analysis/b_run_esol_conformer_diversity.py
```

The complete set of analysis programs is contained in `programs/analysis/`.

---

## Diagnostics and historical programs

Diagnostic programs are located in:

```text
diagnostics/programs/
```

They document additional stability, audit, and forensic analyses associated with the study.

Historical/superseded programs are retained under:

```text
legacy/programs/
```

These programs are preserved for provenance and are not part of the recommended production workflow.

---

## Scientific interpretation

The central result of the benchmark is not that more conformers are universally better, nor that QSARmil universally outperforms 2D QSAR.

The study instead evaluates conformer-aware molecular machine learning as a **representation-design and model-selection problem**.

Within the evaluated QSARmil landscape:

- descriptor identity is the dominant source of performance variation in the ESOL benchmark;
- aggregation/model architecture contributes a smaller overall component, while local descriptor–architecture compatibility can still matter;
- conformer-count performance is non-monotonic rather than universally improving with additional conformers;
- molecule-level improvement is heterogeneous rather than universal;
- dataset behavior differs substantially between ESOL and FreeSolv; and
- repeated-split analysis is important for assessing the stability of conclusions.

The numerical files in this repository preserve the underlying benchmark and analysis results supporting these conclusions.

---

## Data provenance and scope

This repository is a reproducibility package for the computational study. It contains study inputs, generated model outputs, analysis tables, diagnostic outputs, and programs used in the work.

The repository does not claim that every upstream dataset or dependency is newly generated by this study. External software packages remain external dependencies, and the repository preserves the study-specific computational layer built around them.

---

## Citation

When citing this computational package, cite the associated manuscript and the persistent archive DOI assigned to this repository.

The software/reproducibility package is intended to be cited together with the associated manuscript.
