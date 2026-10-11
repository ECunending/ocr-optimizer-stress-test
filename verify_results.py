import json
from pathlib import Path
import sys
import torch
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.resolve()))
from models import Model,initial_matrix
from train import metrics
from data import make_data

def audit():
    paths=sorted((root/'results/runs').glob('*.json'))
    assert len(paths)==50,len(paths)
    worst=0.; crosses=0
    torch.set_num_threads(1)
    for p in paths:
        r=json.loads(p.read_text());assert r['stable']
        payload=torch.load(p.with_suffix('.pt'),weights_only=True)
        for key,state in payload['checkpoints'].items():
            m=Model(r['config']['kind'],initial_matrix(r['config']['seed']));m.load_state_dict(state)
            actual=metrics(m,make_data());record=r['final'] if key=='final' else r['crossings'][key[3:]]
            for metric in actual:
                if actual[metric] is not None: worst=max(worst,abs(actual[metric]-record[metric]))
            assert all(v==record[k] for k,v in actual.items())
            if key!='final':assert actual['train_ce']<=float(key[3:]);crosses+=1
        assert r['history'][-1]['step']==20000
        assert r['final']['train_accuracy']==1.
    for kind in ['direct','factorized']:
        for seed in range(5):
            a=json.loads((root/f'results/runs/{kind}_adam_wd0.0_seed{seed}.json').read_text())
            b=json.loads((root/f'results/runs/{kind}_adamw_wd0.0_seed{seed}.json').read_text())
            assert a['history']==b['history']
    result=dict(runs=50,checkpoint_metrics_exactly_reproduced=True,threshold_checkpoints=crosses,adam_adamw_full_trajectories_identical=True,max_metric_error=worst)
    (root/'results/final_audit.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':audit()
