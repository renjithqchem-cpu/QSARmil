from pathlib import Path
import re
import pandas as pd

BASE = Path.home() / "Documents" / "qsarmil"
OUT = BASE / "data" / "esol_model_audit"
OUT.mkdir(parents=True, exist_ok=True)

CONF_COUNTS = [5, 10, 15, 20]

# Candidate metric names that commonly occur in QSARmil outputs.
METRIC_PATTERNS = {
    "R2": re.compile(r"(?:R2|R²|r2)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I),
    "RMSE": re.compile(r"RMSE\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I),
    "MAE": re.compile(r"MAE\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I),
    "Pearson_r": re.compile(r"(?:Pearson\s*r|pearson_r|Pearson)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I),
}

def conf_from_path(p):
    s = str(p).lower()
    m = re.search(r"(\d+)\s*conf", s)
    if m:
        n = int(m.group(1))
        if n in CONF_COUNTS:
            return n
    return None

def classify_file(p):
    name = p.name.lower()
    if name.endswith(".csv"):
        return "csv"
    if name.endswith((".txt", ".log", ".out")):
        return "text"
    return None

records = []
scanned = []

# Restrict the search to names that are likely related to ESOL/QSARmil runs.
candidates = []
for p in BASE.rglob("*"):
    if not p.is_file():
        continue
    c = conf_from_path(p)
    if c is None:
        continue
    typ = classify_file(p)
    if typ is None:
        continue
    low = str(p).lower()
    if "esol" in low or "qsarmil" in low:
        candidates.append(p)

print("=" * 80)
print("ESOL QSARMIL MODEL AUDIT")
print("=" * 80)
print(f"Base directory: {BASE}")
print(f"Candidate files found: {len(candidates)}")
print()

# First pass: CSV files.
for p in sorted(candidates):
    conf = conf_from_path(p)
    typ = classify_file(p)
    scanned.append((str(p), conf, typ))
    if typ != "csv":
        continue

    try:
        df = pd.read_csv(p)
    except Exception:
        continue

    cols_lower = {str(c).lower(): c for c in df.columns}
    metric_cols = {}
    for metric, aliases in {
        "R2": ["r2", "r²", "r_2", "r2_score"],
        "RMSE": ["rmse"],
        "MAE": ["mae"],
        "Pearson_r": ["pearson_r", "pearson r", "pearson"],
    }.items():
        for alias in aliases:
            if alias in cols_lower:
                metric_cols[metric] = cols_lower[alias]
                break

    # Detect likely descriptor/model columns.
    model_col = next((cols_lower[x] for x in
                      ["model", "estimator", "algorithm", "mil_algorithm",
                       "model_name"] if x in cols_lower), None)
    desc_col = next((cols_lower[x] for x in
                     ["descriptor", "descriptor_name"] if x in cols_lower), None)

    if metric_cols:
        for _, row in df.iterrows():
            rec = {
                "Conformers": conf,
                "Source_file": str(p),
                "Source_type": "csv",
                "Descriptor": row.get(desc_col, "") if desc_col else "",
                "Model": row.get(model_col, "") if model_col else "",
                "R2": row.get(metric_cols.get("R2"), "") if "R2" in metric_cols else "",
                "RMSE": row.get(metric_cols.get("RMSE"), "") if "RMSE" in metric_cols else "",
                "MAE": row.get(metric_cols.get("MAE"), "") if "MAE" in metric_cols else "",
                "Pearson_r": row.get(metric_cols.get("Pearson_r"), "") if "Pearson_r" in metric_cols else "",
            }
            records.append(rec)

# Second pass: text/log files. Extract contextual model names where possible.
model_re = re.compile(
    r"(?P<descriptor>[A-Za-z0-9_]+)\s*\|\s*(?P<model>[A-Za-z0-9_]+)"
)
for p in sorted(candidates):
    if classify_file(p) != "text":
        continue
    conf = conf_from_path(p)
    try:
        text = p.read_text(errors="ignore")
    except Exception:
        continue

    lines = text.splitlines()
    for i, line in enumerate(lines):
        # Only inspect lines containing at least one metric.
        vals = {}
        for metric, pat in METRIC_PATTERNS.items():
            m = pat.search(line)
            if m:
                vals[metric] = float(m.group(1))
        if not vals:
            continue

        context = "\n".join(lines[max(0, i-3):min(len(lines), i+4)])
        mm = model_re.search(context)
        descriptor = mm.group("descriptor") if mm else ""
        model = mm.group("model") if mm else ""

        records.append({
            "Conformers": conf,
            "Source_file": str(p),
            "Source_type": "text",
            "Descriptor": descriptor,
            "Model": model,
            "R2": vals.get("R2", ""),
            "RMSE": vals.get("RMSE", ""),
            "MAE": vals.get("MAE", ""),
            "Pearson_r": vals.get("Pearson_r", ""),
        })

# Save inventory regardless of whether metrics were found.
inventory = pd.DataFrame(scanned, columns=["Source_file", "Conformers", "Source_type"])
inventory.to_csv(OUT / "scanned_file_inventory.csv", index=False)

audit = pd.DataFrame(records)
if not audit.empty:
    audit.to_csv(OUT / "model_level_audit_raw.csv", index=False)

# Try to normalize numeric columns.
for c in ["R2", "RMSE", "MAE", "Pearson_r"]:
    if c in audit.columns:
        audit[c] = pd.to_numeric(audit[c], errors="coerce")

# Create a useful summary if model-level records were recovered.
if not audit.empty:
    usable = audit.dropna(subset=["R2", "RMSE", "MAE"], how="all").copy()

    if not usable.empty:
        # Deduplicate exact duplicate records caused by overlapping text contexts.
        usable = usable.drop_duplicates()

        # Rank only for descriptive auditing; this does NOT select a final model.
        if "RMSE" in usable.columns:
            usable["RMSE_rank_within_conformer"] = usable.groupby("Conformers")["RMSE"].rank(
                method="min", ascending=True
            )
        if "R2" in usable.columns:
            usable["R2_rank_within_conformer"] = usable.groupby("Conformers")["R2"].rank(
                method="min", ascending=False
            )

        usable.to_csv(OUT / "model_level_audit.csv", index=False)

        # Aggregate repeated descriptor/model combinations if identifiable.
        keyed = usable[
            (usable["Descriptor"].astype(str).str.len() > 0) |
            (usable["Model"].astype(str).str.len() > 0)
        ].copy()

        if not keyed.empty:
            agg = keyed.groupby(
                ["Conformers", "Descriptor", "Model"], dropna=False
            ).agg(
                N_records=("RMSE", "count"),
                mean_R2=("R2", "mean"),
                mean_RMSE=("RMSE", "mean"),
                mean_MAE=("MAE", "mean"),
                mean_Pearson_r=("Pearson_r", "mean"),
            ).reset_index()
            agg.to_csv(OUT / "descriptor_model_summary.csv", index=False)

            # Stability across conformer counts for combinations appearing >=2 times.
            stability = agg.groupby(["Descriptor", "Model"]).agg(
                conformer_counts=("Conformers", lambda x: ",".join(map(str, sorted(set(x))))),
                n_conformer_settings=("Conformers", "nunique"),
                mean_RMSE=("mean_RMSE", "mean"),
                mean_MAE=("mean_MAE", "mean"),
                mean_R2=("mean_R2", "mean"),
                sd_RMSE=("mean_RMSE", "std"),
                sd_MAE=("mean_MAE", "std"),
                sd_R2=("mean_R2", "std"),
            ).reset_index()
            stability = stability[stability["n_conformer_settings"] >= 2]
            stability.sort_values(["mean_RMSE", "mean_MAE"], inplace=True)
            stability.to_csv(OUT / "cross_conformer_model_stability.csv", index=False)

# Report.
report = []
report.append("=" * 80)
report.append("ESOL QSARMIL MODEL AUDIT REPORT")
report.append("=" * 80)
report.append(f"Base: {BASE}")
report.append(f"Candidate files scanned: {len(candidates)}")
report.append(f"Model-level records recovered: {len(audit)}")
report.append("")

report.append("SCANNED FILES BY CONFORMER COUNT")
if not inventory.empty:
    report.append(
        inventory.groupby(["Conformers", "Source_type"]).size().reset_index(
            name="N_files"
        ).to_string(index=False)
    )
else:
    report.append("No candidate files found.")
report.append("")

if audit.empty:
    report.append("NO MODEL-LEVEL METRICS WERE AUTOMATICALLY RECOVERED.")
    report.append("This means the existing output format needs a targeted parser.")
    report.append("Do NOT start new expensive QSARmil runs yet.")
else:
    report.append("RECOVERED MODEL-LEVEL RECORDS")
    report.append(audit.to_string(index=False))
    report.append("")
    report.append("If descriptor/model columns are blank, inspect model_level_audit_raw.csv")
    report.append("and the source files before using this table to freeze a model.")

(report_path := OUT / "ESOL_QSARMIL_MODEL_AUDIT_REPORT.txt").write_text(
    "\n".join(report), encoding="utf-8"
)

print("\n".join(report))
print()
print(f"Audit directory: {OUT}")
print(f"Report: {report_path}")
