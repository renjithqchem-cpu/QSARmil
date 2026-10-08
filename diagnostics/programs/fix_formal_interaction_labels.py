from pathlib import Path
import pandas as pd

p=Path('data/descriptor_architecture_formal_interaction/formal_interaction_test_results.csv')
out=p.with_name('formal_interaction_test_results_corrected.csv')
df=pd.read_csv(p)
# Preserve the original values but relabel them correctly.
df['architecture_incremental_SS_corrected']=df['descriptor_incremental_SS']
df['descriptor_incremental_SS_corrected']=df['architecture_incremental_SS']
# Remove ambiguous original labels from the archival corrected table.
df=df.drop(columns=['descriptor_incremental_SS','architecture_incremental_SS'])
df.to_csv(out,index=False)
print(df.to_string(index=False))
print(f'Wrote {out}')
