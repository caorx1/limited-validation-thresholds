"""Verify archived predictions at saved thresholds; never choose a new threshold."""
from pathlib import Path
import argparse, csv, hashlib, json, math
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def macro_f1(y,pred):
    tp=int(np.sum((y==1)&pred));tn=int(np.sum((y==0)&~pred))
    fp=int(np.sum((y==0)&pred));fn=int(np.sum((y==1)&~pred))
    a=2*tp+fp+fn;b=2*tn+fp+fn
    return .5*((2*tp/a if a else 0)+(2*tn/b if b else 0))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'reports/saved_prediction_verification.json');args=ap.parse_args()
    manifest=json.loads((ROOT/'cache_manifest.json').read_text('utf-8'))
    assert manifest['task_count']==45
    source_d=defaultdict(list); paired={}; checks=0;maxerr=0.; taskkeys=set()
    for item in manifest['tasks']:
        path=ROOT/item['path'];assert sha(path)==item['sha256'],f'cache changed: {item["task_id"]}'
        data=json.loads(path.read_text('utf-8'));identity=data['identity']
        key=(identity['source_slug'],identity['model'],identity['outer_seed'])
        assert key not in taskkeys;taskkeys.add(key)
        pc=data['probability_cache'];y=np.asarray(pc['test_labels'],dtype=np.int8);p=np.asarray(pc['test_probabilities'],dtype=float)
        assert len(y)==len(p) and np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1))
        counts=defaultdict(lambda:np.zeros(len(p),dtype=np.int64));nrep=defaultdict(int)
        for row in data['threshold_records']:
            m=row['budget']; rep=row['repeat']; base=macro_f1(y,p>=.5)
            assert row['thresholds']['S0']==.5
            for strategy,t in row['thresholds'].items():
                prediction=p>=t;score=macro_f1(y,prediction)
                err=abs(score-row['test_macro_f1'][strategy]);maxerr=max(maxerr,err);checks+=1
                assert err<=1e-12,(item['task_id'],m,rep,strategy,err)
                counts[(m,strategy)]+=prediction;nrep[(m,strategy)]+=1
                if m in [100,500] and strategy!='S0':paired[key+(m,rep,strategy)]=score-base
        for (m,strategy),pos in counts.items():
            r=nrep[(m,strategy)];assert r==100
            q=pos/r;d=r/(r-1)*float(np.mean(2*q*(1-q)))
            source_d[(key[0],m,strategy)].append(d)
    # Saved primary gains link the newly packaged probability inputs to the manuscript tables.
    gainchecks=0
    with (ROOT/'tables/primary_budget_paired_draws.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==27000 and len(paired)==27000
    for r in rows:
        k=(r['source'],r['model'],int(r['outer_seed']),int(r['budget']),int(r['repeat']),r['strategy'])
        assert abs(paired[k]-float(r['delta']))<=1e-12,k;gainchecks+=1
    dchecks=0;derr=0.
    with (ROOT/'tables/all_strategy_source_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            k=(row['source'],int(row['budget']),row['strategy'])
            assert len(source_d[k])==9
            value=math.fsum(source_d[k])/9;err=abs(value-float(row['decision_instability']))
            derr=max(derr,err);assert err<=1e-12,(k,err);dchecks+=1
    result={'status':'PASS_SAVED_THRESHOLDS_AND_PREDICTIONS_ONLY','task_count':len(taskkeys),'macro_f1_checks':checks,'macro_f1_max_abs_error':maxerr,'primary_paired_gain_checks':gainchecks,'source_decision_disagreement_checks':dchecks,'source_decision_disagreement_max_abs_error':derr,'new_fits':0,'new_threshold_searches':0,'v1_v2_reruns':0,'scope_limit':'Uses archived thresholds; does not verify threshold-search reproduction, training, or sensitivity re-execution.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
