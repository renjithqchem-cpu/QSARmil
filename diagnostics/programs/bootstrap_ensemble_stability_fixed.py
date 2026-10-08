import pandas as pd
import numpy as np
from itertools import combinations

v=pd.read_csv("freesolv_qsarmil_5conf/val.csv")
ms=[c for c in v.columns if c not in ["SMILES","Y_TRUE"]]
y=v.Y_TRUE.to_numpy(float)
P=v[ms].to_numpy(float)
R=P-y[:,None]
n=len(y); m=len(ms); B=2000
rng=np.random.default_rng(42)
idx=rng.integers(0,n,size=(B,n))

pairs=np.array(list(combinations(range(m),2)),dtype=np.int16)
npairs=len(pairs)
sel2=np.empty(B,dtype=np.int32); score2=np.empty(B)

# Exact exhaustive 2-model bootstrap search
for b in range(B):
    rb=R[idx[b]]
    ri=rb[:,pairs[:,0]]+rb[:,pairs[:,1]]
    mse=np.mean((ri*0.5)**2,axis=0)
    j=np.argmin(mse)
    sel2[b]=j; score2[b]=np.sqrt(mse[j])

# 4-model search among the 20 best individual models in each bootstrap
sel4=np.empty(B,dtype=object); score4=np.empty(B)
for b in range(B):
    rb=R[idx[b]]
    indiv=np.sqrt(np.mean(rb**2,axis=0))
    cand=np.argsort(indiv)[:20]
    comb=np.array(list(combinations(cand,4)),dtype=np.int16)
    pred=(rb[:,comb[:,0]]+rb[:,comb[:,1]]+rb[:,comb[:,2]]+rb[:,comb[:,3]])*0.25
    mse=np.mean(pred**2,axis=0)
    j=np.argmin(mse)
    sel4[b]=" || ".join(ms[k] for k in comb[j])
    score4[b]=np.sqrt(mse[j])

pair_names=[" || ".join(ms[k] for k in pairs[j]) for j in sel2]
pair_counts=pd.Series(pair_names).value_counts()
four_counts=pd.Series(sel4).value_counts()

print("="*120)
print("BOOTSTRAP ENSEMBLE-SELECTION STABILITY: 2000 VALIDATION RESAMPLES")
print("="*120)
print("2-MODEL EXHAUSTIVE SEARCH")
print("Mean optimal RMSE:",score2.mean())
print("SD optimal RMSE:",score2.std())
print("Median optimal RMSE:",np.median(score2))
print("Unique optimal pairs:",len(pair_counts))
print("\nTop 20 selected pairs:")
print(pair_counts.head(20).to_string())

print("\n"+"="*120)
print("4-MODEL SEARCH: TOP-20 INDIVIDUAL CANDIDATES PER BOOTSTRAP")
print("="*120)
print("Mean optimal RMSE:",score4.mean())
print("SD optimal RMSE:",score4.std())
print("Median optimal RMSE:",np.median(score4))
print("Unique selected 4-model ensembles:",len(four_counts))
print("\nTop 20 selected 4-model ensembles:")
print(four_counts.head(20).to_string())

sel_models=["MolFeatElectroShape|MeanInstanceWrapperMLPNetworkRegressor","RDKitMORSE|MeanInstanceWrapperMLPNetworkRegressor","MolFeatPmapper|MeanInstanceWrapperMLPNetworkRegressor","RDKitGETAWAY|MeanInstanceWrapperMLPNetworkRegressor"]
print("\n"+"="*120)
print("SELECTION FREQUENCY OF QSARCONS MEMBERS")
print("="*120)
for s in sel_models:
    print("\n",s)
    print("2-model inclusion:",sum(s in x for x in pair_names)/B)
    print("4-model inclusion:",sum(s in x for x in sel4)/B)

print("\n"+"="*120)
print("ORIGINAL QSARCONS ENSEMBLES")
print("="*120)
print("Original k2: MORSE + Pmapper")
print("Original k4: ElectroShape + MORSE + Pmapper + GETAWAY")
