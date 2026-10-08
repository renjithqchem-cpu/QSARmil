from pathlib import Path
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

ROOT=Path('/mnt/data/qsarmil_work/qsarmil/data')

def scaffold_split(df, smiles_col, frac_train=0.8):
    groups={}
    for i,s in enumerate(df[smiles_col].astype(str)):
        m=Chem.MolFromSmiles(s)
        if m is None: raise ValueError(f'Invalid SMILES at row {i}: {s}')
        scaf=MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        groups.setdefault(scaf,[]).append(i)
    ordered=sorted(groups.items(), key=lambda kv:(-len(kv[1]), kv[0]))
    target=int(round(len(df)*frac_train))
    train=[]; test=[]
    for _,idxs in ordered:
        if len(train)+len(idxs)<=target or len(train)==0:
            train.extend(idxs)
        else:
            test.extend(idxs)
    # if train target overshoots/undershoots, keep scaffold integrity and report actual sizes
    if set(train)&set(test): raise RuntimeError('overlap')
    return sorted(train), sorted(test), groups

funcs=[('MolWt',Descriptors.MolWt),('MolLogP',Crippen.MolLogP),('TPSA',rdMolDescriptors.CalcTPSA),('HBD',Lipinski.NumHDonors),('HBA',Lipinski.NumHAcceptors),('RotB',Lipinski.NumRotatableBonds),('RingCount',Lipinski.RingCount),('AromaticRings',Lipinski.NumAromaticRings),('HeavyAtomCount',lambda m:m.GetNumHeavyAtoms()),('FractionCSP3',rdMolDescriptors.CalcFractionCSP3)]

def X(smiles):
    out=[]
    for s in smiles:
        m=Chem.MolFromSmiles(s); out.append([f(m) for _,f in funcs])
    return np.asarray(out,float)

def metrics(y,p):
    return dict(R2=r2_score(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),MAE=mean_absolute_error(y,p),Pearson_r=pearsonr(y,p)[0],Spearman_rho=spearmanr(y,p)[0])

cases=[('ESOL',ROOT/'delaney-processed.csv','smiles','measured log solubility in mols per litre'),('FreeSolv',ROOT/'freesolv_benchmark.csv','smiles','experimental_dG_hyd_kcal_mol')]
rows=[]
for name,path,sc,ycol in cases:
    df=pd.read_csv(path)
    tr,te,groups=scaffold_split(df,sc)
    train=df.iloc[tr].copy(); test=df.iloc[te].copy()
    Xtr=X(train[sc]); Xte=X(test[sc]); ytr=train[ycol].to_numpy(float); yte=test[ycol].to_numpy(float)
    for model_name,model in [
        ('RandomForest',RandomForestRegressor(n_estimators=500,random_state=42,n_jobs=-1,max_features='sqrt')),
        ('ExtraTrees',ExtraTreesRegressor(n_estimators=500,random_state=42,n_jobs=-1,max_features=1.0))]:
        model.fit(Xtr,ytr); p=model.predict(Xte); mm=metrics(yte,p)
        rows.append({'dataset':name,'model':model_name,'N_total':len(df),'N_train':len(train),'N_test':len(test),'N_scaffolds_total':len(groups),'N_scaffolds_train':len(set(k for k,v in groups.items() if set(v)<=set(tr))),'N_scaffolds_test':len(set(k for k,v in groups.items() if set(v)<=set(te))),**mm})
        test[['smiles',ycol]].assign(prediction=p).to_csv(f'/mnt/data/{name.lower()}_scaffold_{model_name.lower()}_predictions.csv',index=False)
res=pd.DataFrame(rows); res.to_csv('/mnt/data/scaffold_2d_sensitivity_summary.csv',index=False); print(res.to_string(index=False))
