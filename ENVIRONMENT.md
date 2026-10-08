# Validated computational environment

The following versions were recorded directly from the validated `qsarmil` environment used for the reported calculations.

- Python: 3.11.16
- qsarmil: 1.0.8
- qsarcons: 1.1.6
- scikit-learn: 1.9.1
- RDKit: 2026.03.6
- molfeat: 1.0.0
- mikit-learn: 1.0.6
- NumPy: 1.26.4
- SciPy: 1.17.1
- pandas: environment snapshot retained in `requirements-lock.txt`
- matplotlib: 3.10.9
- datamol: 0.13.0
- pmapper: 1.1.3
- XGBoost: 3.2.0
- PyTorch: retained in `requirements-lock.txt`
- PyTorch Lightning: 2.6.6
- statsmodels: retained in `requirements-lock.txt`

`requirements-lock.txt` is the exact `pip freeze` snapshot exported from the validated environment. Some entries contain local conda-build file URLs because the environment was assembled from conda/rattler packages. Those entries are retained deliberately as provenance and should not be interpreted as universally portable pip-install commands.

The study-specific scripts use the installed upstream `qsarmil` and `qsarcons` packages; their source code is not duplicated in this repository.
