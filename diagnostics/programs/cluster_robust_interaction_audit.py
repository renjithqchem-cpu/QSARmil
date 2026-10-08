from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import f

ROOT = Path('/mnt/data/qsarmil_work/qsarmil/data')
IN = ROOT/'descriptor_architecture_formal_interaction'
OUT = ROOT/'descriptor_architecture_robust_inference'
OUT.mkdir(exist_ok=True)

DATASETS={
 'ESOL': IN/'ESOL_esol_qsarmil_20conf_molecule_absolute_errors.csv',
 'FreeSolv': IN/'FreeSolv_5conf_molecule_absolute_errors.csv'
}

rows=[]
for name,path in DATASETS.items():
    df=pd.read_csv(path)
    df['molecule']=df['molecule'].astype(str)
    df['descriptor']=df['descriptor'].astype('category')
    df['architecture']=df['architecture'].astype('category')
    formula='absolute_error ~ C(molecule) + C(descriptor) * C(architecture)'
    model=smf.ols(formula,data=df).fit(cov_type='cluster', cov_kwds={'groups':df['molecule']})
    params=list(model.params.index)
    idx=[i for i,p in enumerate(params) if 'C(descriptor)' in p and 'C(architecture)' in p and ':' in p]
    R=np.zeros((len(idx),len(params)))
    for j,i in enumerate(idx): R[j,i]=1.0
    wt=model.wald_test(R, use_f=True)
    stat=float(np.asarray(wt.statistic).squeeze())
    p=float(np.asarray(wt.pvalue).squeeze())
    df_num=int(getattr(wt,'df_num',len(idx)))
    df_den=float(getattr(wt,'df_denom',np.nan))
    # conventional partial eta2 retained from unclustered SS decomposition as an effect-size descriptor
    red=smf.ols('absolute_error ~ C(molecule) + C(descriptor) + C(architecture)',data=df).fit()
    ssr_red=np.sum(red.resid**2); ssr_full=np.sum(model.resid**2)
    ss_int=ssr_red-ssr_full
    eta=ss_int/(ss_int+ssr_full)
    rows.append({'dataset':name,'n_molecules':df.molecule.nunique(),'n_obs':len(df),'interaction_terms':len(idx),'cluster_robust_F':stat,'cluster_robust_p':p,'cluster_df_num':df_num,'cluster_df_denom':df_den,'partial_eta2_unclustered':eta,'OLS_nominal_F':((ss_int/len(idx))/(ssr_full/model.df_resid)),'OLS_nominal_p':f.sf((ss_int/len(idx))/(ssr_full/model.df_resid),len(idx),model.df_resid)})
    # save parameter table for audit
    pd.DataFrame({'parameter':params,'coef':model.params.values,'cluster_se':model.bse.values,'cluster_p':model.pvalues.values}).to_csv(OUT/f'{name}_cluster_robust_coefficients.csv',index=False)
res=pd.DataFrame(rows)
res.to_csv(OUT/'cluster_robust_interaction_summary.csv',index=False)
print(res.to_string(index=False))
