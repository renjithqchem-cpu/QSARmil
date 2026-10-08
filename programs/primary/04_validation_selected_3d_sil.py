from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from qsarmil.modelling import lazy

SEED=42
N_CONFORMERS=20


def pearson(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    if len(y)<2 or np.std(y)==0 or np.std(p)==0: return np.nan
    return float(np.corrcoef(y,p)[0,1])


def lowest_uff(smiles, num_conf=N_CONFORMERS, seed=SEED):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return None,np.nan,np.nan,-1
    mh=Chem.AddHs(mol)
    prm=AllChem.ETKDGv3(); prm.randomSeed=seed; prm.numThreads=1
    ids=list(AllChem.EmbedMultipleConfs(mh,numConfs=num_conf,params=prm))
    if not ids: return None,np.nan,np.nan,-1
    energies=[]
    for cid in ids:
        try:
            AllChem.UFFOptimizeMolecule(mh,confId=cid,maxIters=2000)
            ff=AllChem.UFFGetMoleculeForceField(mh,confId=cid)
            e=float(ff.CalcEnergy())
            if np.isfinite(e): energies.append((cid,e))
        except Exception: pass
    if not energies: return None,np.nan,np.nan,-1
    energies.sort(key=lambda x:x[1])
    best,e0=energies[0]; emax=energies[-1][1]
    selected=Chem.Mol(mh); selected.RemoveAllConformers()
    selected.AddConformer(mh.GetConformer(best),assignId=True)
    selected=Chem.RemoveHs(selected)
    return selected,e0,emax,best


def load_case(case, root):
    if case=='ESOL':
        data=root/'data/delaney-processed.csv'
        train_file=root/'data/esol_train.csv'
        test_file=root/'data/esol_test.csv'
        target='measured log solubility in mols per litre'
    else:
        data=root/'data/freesolv_benchmark.csv'
        train_file=root/'data/freesolv_train_seed42.csv'
        test_file=root/'data/freesolv_test_seed42.csv'
        target='experimental_dG_hyd_kcal_mol'
    df=pd.read_csv(data); tr=pd.read_csv(train_file); te=pd.read_csv(test_file)
    smiles=df['smiles'].astype(str).tolist(); y=df[target].astype(float).to_numpy()
    train_set=set(tr['smiles'].astype(str)); test_set=set(te['smiles'].astype(str))
    train_idx=[i for i,s in enumerate(smiles) if s in train_set]
    test_idx=[i for i,s in enumerate(smiles) if s in test_set]
    assert len(train_idx)==len(tr) and len(test_idx)==len(te) and train_set.isdisjoint(test_set)
    return df,target,smiles,y,train_idx,test_idx


def run_case(case, root, out):
    df,target,smiles,y,train_idx,test_idx=load_case(case,root)
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    case_out=out/case; case_out.mkdir(exist_ok=True)
    print(f'\n===== {case}: leakage-free 3D-SIL selection =====')
    print('Total/train/test:',len(df),len(train_idx),len(test_idx))

    mols=[]; emin=[]; emax=[]; cid=[]
    for i,s in enumerate(smiles):
        m,e0,ee,c=lowest_uff(s)
        mols.append(m); emin.append(e0); emax.append(ee); cid.append(c)
        if (i+1)%50==0 or i==len(smiles)-1: print(f'conformers {i+1}/{len(smiles)}')

    train_valid=[i for i in train_idx if mols[i] is not None]
    test_valid=[i for i in test_idx if mols[i] is not None]
    if len(train_valid)!=len(train_idx) or len(test_valid)!=len(test_idx):
        raise RuntimeError('Conformer generation failure changed the frozen split; inspect failures before proceeding.')

    tr_local=np.arange(len(train_valid))
    fit_local,val_local=train_test_split(tr_local,test_size=0.20,random_state=SEED)
    fit_local=list(fit_local); val_local=list(val_local)
    mol_fit=[mols[i] for i in [train_valid[j] for j in fit_local]]
    mol_val=[mols[i] for i in [train_valid[j] for j in val_local]]
    mol_test=[mols[i] for i in test_valid]
    y_fit=y[[train_valid[j] for j in fit_local]]
    y_val=y[[train_valid[j] for j in val_local]]
    y_test=y[test_valid]
    smi_test=[smiles[i] for i in test_valid]

    candidates=[]; test_preds={}
    for dname,dfactory in lazy.DESCRIPTORS.items():
        print('Descriptor:',dname)
        try:
            calc=dfactory()
            bags=[[m] for m in (mol_fit+mol_val+mol_test)]
            X=lazy.calculate_descriptors(bags,calc)
            nf,nv=len(mol_fit),len(mol_val)
            Xf,Xv,Xt=X[:nf],X[nf:nf+nv],X[nf+nv:]
        except Exception as exc:
            for ename in lazy.REGRESSORS:
                candidates.append({'descriptor':dname,'architecture':ename,'model':f'{dname}|{ename}','status':'DESCRIPTOR_FAILED','validation_RMSE':np.nan,'error':repr(exc)})
            continue
        for ename,efactory in lazy.REGRESSORS.items():
            model=f'{dname}|{ename}'
            try:
                est=efactory(accelerator='cpu')
                # IMPORTANT: no package HPO here. Outer validation is the selection set.
                _,pv,pt=lazy.train_estimator(Xf,Xv,Xt,y_fit,y_val,est,False,random_seed=SEED,accelerator='cpu')
                pv=np.asarray(pv,float); pt=np.asarray(pt,float)
                vrm=float(np.sqrt(mean_squared_error(y_val,pv)))
                vma=float(mean_absolute_error(y_val,pv))
                candidates.append({'descriptor':dname,'architecture':ename,'model':model,'status':'OK','validation_RMSE':vrm,'validation_MAE':vma,'validation_R2':float(r2_score(y_val,pv)),'validation_Pearson':pearson(y_val,pv)})
                test_preds[model]=pt
            except Exception as exc:
                candidates.append({'descriptor':dname,'architecture':ename,'model':model,'status':'FAILED','validation_RMSE':np.nan,'error':repr(exc)})

    cand=pd.DataFrame(candidates); cand.to_csv(case_out/'3d_sil_validation_candidate_metrics.csv',index=False)
    ok=cand[cand.status=='OK'].sort_values(['validation_RMSE','validation_MAE','model']).reset_index(drop=True)
    if ok.empty: raise RuntimeError(f'{case}: no valid 3D-SIL candidate models')
    selected=ok.iloc[0]
    model=selected['model']
    ptest=np.asarray(test_preds[model],float)
    result={'dataset':case,'N_total':len(df),'N_train':len(train_idx),'N_fit':len(fit_local),'N_validation':len(val_local),'N_test':len(test_idx),'selected_model':model,'selected_descriptor':selected['descriptor'],'selected_architecture':selected['architecture'],'validation_RMSE':selected['validation_RMSE'],'validation_MAE':selected.get('validation_MAE',np.nan),'validation_R2':selected.get('validation_R2',np.nan),'test_R2':float(r2_score(y_test,ptest)),'test_RMSE':float(np.sqrt(mean_squared_error(y_test,ptest))),'test_MAE':float(mean_absolute_error(y_test,ptest)),'test_Pearson_r':pearson(y_test,ptest)}
    pd.DataFrame([result]).to_csv(case_out/'3d_sil_selected_result.csv',index=False)
    pd.DataFrame({'SMILES':smi_test,'Y_TRUE':y_test,'prediction':ptest}).to_csv(case_out/'3d_sil_selected_test_predictions.csv',index=False)
    pd.DataFrame({'SMILES':smiles,'target':y,'split':['train' if i in train_idx else 'test' for i in range(len(smiles))],'UFF_selected_energy_kcal_mol':emin,'UFF_max_energy_kcal_mol':emax,'selected_conformer':cid}).to_csv(case_out/'selected_conformer_info.csv',index=False)
    print('SELECTED:',model)
    print('Validation RMSE:',result['validation_RMSE'])
    print('TEST:',result['test_R2'],result['test_RMSE'],result['test_MAE'],result['test_Pearson_r'])
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--case',choices=['ESOL','FreeSolv','both'],default='both'); ap.add_argument('--root',default='.') ; ap.add_argument('--out',default='data/validation_selected_3d_sil')
    a=ap.parse_args(); cases=['ESOL','FreeSolv'] if a.case=='both' else [a.case]
    res=[run_case(c,Path(a.root),a.out) for c in cases]
    pd.DataFrame(res).to_csv(Path(a.out)/'validation_selected_3d_sil_summary.csv',index=False)
