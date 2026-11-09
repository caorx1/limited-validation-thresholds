from pathlib import Path
import json, platform
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'tables/primary_budget_paired_draws.csv')
ref=pd.read_csv(ROOT/'tables/all_strategy_equal_source_metrics.csv')
source_ref=pd.read_csv(ROOT/'tables/all_strategy_source_metrics.csv')
assert len(d)==27000 and not d.duplicated(['source','model','outer_seed','budget','repeat','strategy']).any()
rows=[]
checked=0
for (m,st),z in d.groupby(['budget','strategy']):
    assert z.groupby(['source','model','outer_seed']).size().eq(100).all()
    z=z.assign(negative=(z.delta<0).astype(float),large=(z.delta<-.01).astype(float))
    task=z.groupby(['source','model','outer_seed'])[['delta','negative','large']].mean()
    model=task.groupby(['source','model']).mean()
    source=model.groupby('source').mean()
    val=source.mean();old=ref[(ref.budget==m)&(ref.strategy==st)].iloc[0]
    for a,b in [('delta','mean_delta_vs_s0'),('negative','negative_draw_fraction'),('large','drop_gt_01_fraction')]:
        assert abs(val[a]-old[b])<1e-12,(m,st,a)
        checked+=1
        for slug,sr in source.iterrows():
            saved=source_ref[(source_ref.source==slug)&(source_ref.budget==m)&(source_ref.strategy==st)].iloc[0]
            assert abs(sr[a]-saved[b])<1e-12,(slug,m,st,a)
            checked+=1
    rows.append({'budget':int(m),'strategy':st,**val.to_dict()})
pd.DataFrame(rows).to_csv(ROOT/'rebuilt_primary_summary.csv',index=False)
(ROOT/'rebuild_check.json').write_text(json.dumps({'method_budget_cells':6,'numeric_comparisons':checked,'tolerance':1e-12,'new_model_fits':0,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},indent=2))
print('Six method-budget cells and source summaries reconciled in '+str(checked)+' comparisons; no fitting or new samples.')
