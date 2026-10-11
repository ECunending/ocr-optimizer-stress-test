"""Training-only pilot followed by paired five-seed optimizer/decay experiments."""
import argparse
import json
import platform
from pathlib import Path
import torch
from train import run
from data import make_data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/experiment.json')
    p.add_argument('--results',default='results')
    args=p.parse_args(); c=json.loads(Path(args.config).read_text()); root=Path(args.results)
    root.mkdir(parents=True,exist_ok=True)
    x,y,xt,yt=make_data()
    dataset=dict(data_seed=1729,n_groups=20,subjects_per_group=4,observed_implications_per_group=1,
                 train=[dict(subject=int(v.nonzero()[0]),relation=int(v.nonzero()[1]),answer=int(a)) for v,a in zip(x,y)],
                 test=[dict(subject=int(v.nonzero()[0]),relation=int(v.nonzero()[1]),answer=int(a)) for v,a in zip(xt,yt)])
    (root/'dataset.json').write_text(json.dumps(dataset,indent=2))
    selected={}; pilot=[]
    for kind in ['direct','factorized']:
        for opt,rates in c['learning_rates'].items():
            candidates=[]
            for lr in rates:
                path=root/'pilot'/f'{kind}_{opt}_{lr}.json'
                r=run(kind,opt,lr,seed=c['pilot_seed'],steps=c['pilot_steps'],evaluate_test=False,output=path)
                pilot.append(r); candidates.append(r)
                print('pilot',kind,opt,lr,r['final'],flush=True)
            stable=[r for r in candidates if r['stable']]
            if not stable: raise RuntimeError('No stable pilot candidate')
            best=min(stable,key=lambda r:r['final']['train_ce'])
            selected[f'{kind}_{opt}']=best['config']['lr']
    (root/'selection.json').write_text(json.dumps(dict(config=c,learning_rates=selected),indent=2))
    (root/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,platform=platform.platform(),threads=1,dtype='float64',device='cpu'),indent=2))
    for kind in ['direct','factorized']:
        for opt,wd in [('gd',0.),('adam',0.),('adamw',0.)]+[('adamw',v) for v in c['weight_decays']]:
            lr=selected[f'{kind}_{"adam" if opt=="adamw" else opt}']
            for seed in c['seeds']:
                path=root/'runs'/f'{kind}_{opt}_wd{wd}_seed{seed}.json'
                r=run(kind,opt,lr,wd=wd,seed=seed,steps=c['steps'],thresholds=c['thresholds'],output=path)
                print('run',path.stem,r['final']['train_ce'],r['final']['test_accuracy'],flush=True)


if __name__=='__main__': main()
