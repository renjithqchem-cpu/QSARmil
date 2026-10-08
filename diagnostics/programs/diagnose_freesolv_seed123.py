from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem
from qsarmil.modelling import lazy
from qsarmil.modelling.meta import MultiConformerRegressor

SEED = 123
INNER_SEED = 42
NCONF = 5

def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y,p))),
        "MAE": float(mean_absolute_error(y,p)),
        "R2": float(r2_score(y,p)),
    }

def make_conformer_instances(smiles, nconf=NCONF):
    bags = []
    failed = []
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            failed.append((idx, smi, "invalid SMILES")); continue
        mh = Chem.AddHs(mol)
        prm = AllChem.ETKDGv3(); prm.randomSeed = 42; prm.numThreads = 1
        cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=nconf, params=prm))
        if not cids:
            failed.append((idx, smi, "embedding failed")); continue
        inst = []
        for cid in cids:
            try:
                AllChem.UFFOptimizeMolecule(mh, confId=cid, maxIters=2000)
            except Exception:
                pass
            one = Chem.Mol(mh)
            conf = Chem.Conformer(mh.GetConformer(cid), True)
            one.RemoveAllConformers(); one.AddConformer(conf, assignId=True)
            inst.append(one)
        bags.append(inst)
    return bags, failed

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--out',default='data/freesolv_seed123_diagnostic')
    args=ap.parse_args()
    root=Path(args.root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    data=root/'data'/'freesolv_benchmark.csv'
    df=pd.read_csv(data)
    smiles=df['smiles'].astype(str).to_numpy()
    y=df['experimental_dG_hyd_kcal_mol'].astype(float).to_numpy()
    idx=np.arange(len(df))
    train_idx,test_idx=train_test_split(idx,test_size=.20,random_state=SEED)
    train_idx=np.sort(train_idx); test_idx=np.sort(test_idx)
    pd.DataFrame({'row_index':idx,'SMILES':smiles,'target':y,
                  'split':np.where(np.isin(idx,test_idx),'test','train')}).to_csv(out/'outer_split_seed123.csv',index=False)

    smiles_train=smiles[train_idx]; y_train=y[train_idx]
    smiles_test=smiles[test_idx]; y_test=y[test_idx]

    model=MultiConformerRegressor(num_conf=NCONF,hopt=False,verbose=True,
                                  random_seed=42,accelerator='cpu',
                                  output_folder=str(out/'qsarmil'))
    pred=model.train_predict(list(smiles_train),y_train,list(smiles_test))
    pred=np.asarray(pred,float)
    selected=list(getattr(model,'best_consensus',[]))
    (out/'selected_consensus.txt').write_text('\n'.join(selected)+'\n',encoding='utf-8')

    test_path=out/'qsarmil'/'test.csv'
    val_path=out/'qsarmil'/'val.csv'
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    test=pd.read_csv(test_path)
    val=pd.read_csv(val_path) if val_path.exists() else None
    ycol='Y_TRUE' if 'Y_TRUE' in test.columns else test.columns[1]
    model_cols=[c for c in test.columns if c not in {test.columns[0],ycol}]
    rows=[]
    for c in model_cols:
        m=metrics(test[ycol],test[c])
        rows.append({'model':c,**m,'min_prediction':float(test[c].min()),'max_prediction':float(test[c].max()),
                     'n_abs_prediction_gt_20':int((test[c].abs()>20).sum()),
                     'n_abs_prediction_gt_100':int((test[c].abs()>100).sum())})
    pd.DataFrame(rows).sort_values('RMSE').to_csv(out/'all_individual_test_model_metrics.csv',index=False)

    selected_rows=[]
    for c in selected:
        if c in test.columns:
            selected_rows.append({'model':c,**metrics(test[ycol],test[c]),
                                  'min_prediction':float(test[c].min()),'max_prediction':float(test[c].max()),
                                  'n_abs_prediction_gt_20':int((test[c].abs()>20).sum())})
    pd.DataFrame(selected_rows).to_csv(out/'selected_model_test_metrics.csv',index=False)

    # Reconstruct the arithmetic consensus directly from the saved selected columns.
    if selected_rows:
        available=[c for c in selected if c in test.columns]
        consensus=test[available].mean(axis=1).to_numpy()
        forensic=pd.DataFrame({'SMILES':test.iloc[:,0].astype(str),'Y_TRUE':test[ycol].to_numpy(),
                               'consensus_prediction':consensus})
        for c in available: forensic[c]=test[c].to_numpy()
        forensic['abs_error_consensus']=np.abs(forensic.Y_TRUE-forensic.consensus_prediction)
        forensic['consensus_sq_error']=(forensic.Y_TRUE-forensic.consensus_prediction)**2
        forensic.sort_values('consensus_sq_error',ascending=False).to_csv(out/'selected_consensus_molecule_forensics.csv',index=False)
        print('Reconstructed consensus metrics:',metrics(forensic.Y_TRUE,forensic.consensus_prediction))

    # Raw RDF descriptor audit: compare outer training vs test conformer instances.
    # This is a representation diagnostic, not a replacement for the QSARmil model.
    train_bags,fail_tr=make_conformer_instances(smiles_train)
    test_bags,fail_te=make_conformer_instances(smiles_test)
    rdf_factory=lazy.DESCRIPTORS['RDKitRDF']
    rdf_calc=rdf_factory()
    train_X=lazy.calculate_descriptors(train_bags,rdf_calc)
    test_X=lazy.calculate_descriptors(test_bags,rdf_calc)
    np.savez_compressed(out/'raw_RDF_descriptor_audit.npz',train=np.asarray(train_X,dtype=object),test=np.asarray(test_X,dtype=object))
    (out/'conformer_generation_failures.txt').write_text(
        'TRAIN FAILURES\n'+repr(fail_tr)+'\nTEST FAILURES\n'+repr(fail_te)+'\n',encoding='utf-8')

    # Summarize finite ranges per descriptor component over all training conformers.
    def flatten(bags):
        arr=[]
        for bag in bags:
            for x in bag:
                arr.append(np.asarray(x,float))
        return np.vstack(arr)
    # Depending on QSARmil's descriptor container, X may be a list/array of bag matrices.
    def flatten_X(X):
        parts=[]
        for x in X:
            a=np.asarray(x,float)
            if a.ndim==1: parts.append(a[None,:])
            else: parts.append(a.reshape(-1,a.shape[-1]))
        return np.vstack(parts)
    A=flatten_X(train_X); B=flatten_X(test_X)
    finite=np.isfinite(A)
    med=np.nanmedian(np.where(finite,A,np.nan),axis=0)
    q1=np.nanpercentile(np.where(finite,A,np.nan),25,axis=0)
    q3=np.nanpercentile(np.where(finite,A,np.nan),75,axis=0)
    iqr=q3-q1
    scale=np.where(iqr>0,iqr,1.0)
    summary=[]
    for j in range(A.shape[1]):
        summary.append({'descriptor_index':j,'train_min':np.nanmin(A[:,j]),'train_max':np.nanmax(A[:,j]),
                        'test_min':np.nanmin(B[:,j]),'test_max':np.nanmax(B[:,j]),
                        'test_abs_exceeds_train_max':int((np.abs(B[:,j])>np.abs(A[:,j]).max()).sum())})
    pd.DataFrame(summary).to_csv(out/'raw_RDF_component_ranges.csv',index=False)
    print('Saved diagnostic outputs to',out)
    print('Selected consensus:',selected)

if __name__=='__main__': main()

