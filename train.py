"""Full-batch CPU training; exact first-crossing checkpoints, no early stopping."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from data import make_data
from models import Model, initial_matrix


def optimizer_for(model, name, lr, wd):
    if name == 'gd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=wd)
    cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[name]
    return cls(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8, foreach=False)


def margin(logits, y):
    other = logits.clone()
    other[torch.arange(len(y)), y] = -torch.inf
    return logits[torch.arange(len(y)), y] - other.max(dim=1).values


@torch.no_grad()
def metrics(model, data):
    x, y, xt, yt = data
    w = model.matrix()
    wc = w - w.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(wc)
    train, test = model(x), model(xt)
    tm, hm = margin(train, y), margin(test, yt)
    gamma = tm.min().item()
    fro, nuclear = torch.linalg.vector_norm(sv).item(), sv.sum().item()
    return dict(train_ce=F.cross_entropy(train,y).item(), train_accuracy=(train.argmax(1)==y).double().mean().item(),
                test_ce=F.cross_entropy(test,yt).item(), test_accuracy=(test.argmax(1)==yt).double().mean().item(),
                train_margin_min=gamma, test_margin_mean=hm.mean().item(), test_margin_min=hm.min().item(),
                centered_frobenius=fro, centered_nuclear=nuclear, spectral_ratio=nuclear/fro,
                spectral_dimension=(nuclear/fro)**2,
                normalized_nuclear=nuclear/gamma if gamma>0 else None,
                normalized_frobenius=fro/gamma if gamma>0 else None)


def run(kind, optimizer, lr, wd=0., seed=0, steps=20000, thresholds=(0.1,0.03,0.01),
        log_every=100, output=None, evaluate_test=True):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = make_data()
    x,y,xt,yt = data
    w0 = initial_matrix(seed)
    model = Model(kind, w0)
    error = (Model('direct',w0)(torch.cat((x,xt))) - Model('factorized',w0)(torch.cat((x,xt)))).abs().max().item()
    assert error < 1e-6
    opt = optimizer_for(model, optimizer, lr, wd)
    config = dict(kind=kind,optimizer=optimizer,lr=lr,wd=wd,seed=seed,steps=steps,thresholds=list(thresholds),log_every=log_every)
    records, crossed, states = [], {}, {}
    start = time.perf_counter()
    stable = True
    for step in range(steps+1):
        logits = model(x)
        loss = F.cross_entropy(logits,y)
        ce = loss.item()
        if not torch.isfinite(loss):
            stable=False
            break
        new = [t for t in thresholds if str(t) not in crossed and ce <= t]
        if evaluate_test and (step % log_every == 0 or step == steps or new):
            row = dict(step=step, **metrics(model,data))
            records.append(row)
            for t in new:
                crossed[str(t)] = row
                states['ce_'+str(t)] = {k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            for t in new:
                crossed[str(t)] = dict(step=step, train_ce=ce)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    final = metrics(model,data) if evaluate_test else dict(train_ce=ce, train_accuracy=(logits.argmax(1)==y).double().mean().item())
    result = dict(config=config,initial_logit_error=error,stable=stable,elapsed_seconds=time.perf_counter()-start,
                  crossings=crossed,final=final,history=records)
    if output:
        p=Path(output); p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2,allow_nan=False))
        if evaluate_test:
            states['final'] = {k:v.detach().clone() for k,v in model.state_dict().items()}
            torch.save(dict(config=config,checkpoints=states),p.with_suffix('.pt'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['direct','factorized'],default='factorized')
    p.add_argument('--optimizer',choices=['gd','adam','adamw'],default='gd')
    p.add_argument('--lr',type=float,default=1.)
    p.add_argument('--wd',type=float,default=0.)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--output',default='results/single.json')
    print(json.dumps(run(**vars(p.parse_args()))['final'],indent=2))
