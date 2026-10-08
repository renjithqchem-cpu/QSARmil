from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

BASE=Path.home()/"Documents"/"qsarmil"
OUT=BASE/"data"/"esol_72model_audit"
OUT.mkdir(parents=True,exist_ok=True)
confs=[5,10,15,20]
rows=[]

for conf in confs:
    f=BASE/"data"/f"esol_qsarmil_{conf}conf"/"test.csv"
    print(f"\nReading: {f}")
    if not f.exists():
        print("  MISSING")
        continue
    d=pd.read_csv(f)
    models=[c for c in d.columns if "|" in str(c)]
    print(f"  rows={len(d)}, model columns={len(models)}")
    y=pd.to_numeric(d["Y_TRUE"],errors="coerce").to_numpy()
    for m in models:
        pred=pd.to_numeric(d[m],errors="coerce").to_numpy()
        ok=np.isfinite(y)&np.isfinite(pred)
        yy,pp=y[ok],pred[ok]
        r,p=pearsonr(yy,pp) if len(yy)>1 else (np.nan,np.nan)
        desc,alg=str(m).split("|",1)
        rows.append(dict(Conformers=conf,Descriptor=desc,Model=alg,Model_full=m,
                         N=len(yy),R2=r2_score(yy,pp),
                         RMSE=np.sqrt(mean_squared_error(yy,pp)),
                         MAE=mean_absolute_error(yy,pp),Pearson_r=r,Pearson_p=p))

a=pd.DataFrame(rows)
if a.empty: raise RuntimeError("No model predictions recovered.")
a.to_csv(OUT/"all_72_models.csv",index=False)

best=[]
for c in confs:
    s=a[a.Conformers==c]
    for criterion,idx in [("RMSE",s.RMSE.idxmin()),("MAE",s.MAE.idxmin()),("R2",s.R2.idxmax())]:
        x=s.loc[idx]
        best.append(dict(Conformers=c,Criterion=criterion,Descriptor=x.Descriptor,
                         Model=x.Model,Model_full=x.Model_full,R2=x.R2,RMSE=x.RMSE,
                         MAE=x.MAE,Pearson_r=x.Pearson_r))
pd.DataFrame(best).to_csv(OUT/"best_models_by_conformer.csv",index=False)

a["RMSE_rank"]=a.groupby("Conformers").RMSE.rank(method="min")
a["MAE_rank"]=a.groupby("Conformers").MAE.rank(method="min")
a["R2_rank"]=a.groupby("Conformers").R2.rank(method="min",ascending=False)
a["Mean_rank"]=a[["RMSE_rank","MAE_rank","R2_rank"]].mean(axis=1)
a.sort_values(["Conformers","RMSE"]).to_csv(OUT/"all_72_models_ranked.csv",index=False)

st=a.pivot_table(index=["Descriptor","Model","Model_full"],columns="Conformers",
                 values=["R2","RMSE","MAE","Pearson_r"],aggfunc="first")
st.columns=[f"{x}_{y}conf" for x,y in st.columns]
st=st.reset_index()
st.to_csv(OUT/"model_stability_across_conformers.csv",index=False)

top=[]
for c in confs:
    s=a[a.Conformers==c].sort_values("RMSE").head(10)
    for rank,(_,x) in enumerate(s.iterrows(),1):
        top.append(dict(Conformers=c,RMSE_rank=rank,Descriptor=x.Descriptor,
                        Model=x.Model,Model_full=x.Model_full,R2=x.R2,RMSE=x.RMSE,
                        MAE=x.MAE,Pearson_r=x.Pearson_r))
top=pd.DataFrame(top)
top.to_csv(OUT/"top10_RMSE_each_conformer.csv",index=False)
counts=top.groupby(["Descriptor","Model","Model_full"]).agg(
    top10_appearances=("Conformers","count"),
    conformer_settings=("Conformers",lambda x: ",".join(map(str,sorted(x))))
).reset_index().sort_values("top10_appearances",ascending=False)
counts.to_csv(OUT/"top10_model_appearance_counts.csv",index=False)

summary=a.groupby(["Descriptor","Model","Model_full"]).agg(
    conformer_settings=("Conformers","nunique"),mean_RMSE=("RMSE","mean"),
    median_RMSE=("RMSE","median"),sd_RMSE=("RMSE","std"),
    mean_MAE=("MAE","mean"),median_MAE=("MAE","median"),
    mean_R2=("R2","mean"),median_R2=("R2","median"),
    mean_Pearson_r=("Pearson_r","mean")).reset_index().sort_values("mean_RMSE")
summary.to_csv(OUT/"model_cross_conformer_summary.csv",index=False)

print("\n"+"="*80)
print("ESOL QSARMIL 72-MODEL AUDIT COMPLETE")
print("="*80)
print(f"Output: {OUT}")
print("\nRECORDS:")
print(a.groupby("Conformers").size())
print("\nBEST MODELS:")
print(pd.DataFrame(best).to_string(index=False))
print("\nTOP-10 APPEARANCE:")
print(counts.head(20).to_string(index=False))
print(f"\nReport files written to: {OUT}")
