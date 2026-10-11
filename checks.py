"""Scientific invariants and optimizer equivalence checks."""
import json
from pathlib import Path
import torch
from data import make_data
from models import Model, initial_matrix
from train import optimizer_for, margin

def main():
    torch.set_num_threads(1)
    x,y,xt,yt=make_data()
    assert x.shape==(100,82) and xt.shape==(60,82)
    assert torch.all(x.sum(1)==2) and torch.all(xt.sum(1)==2)
    assert not set(map(tuple,x.tolist())) & set(map(tuple,xt.tolist()))
    assert torch.equal(torch.bincount(y,minlength=40),torch.tensor([4]*20+[1]*20))
    assert torch.equal(torch.bincount(yt,minlength=40),torch.tensor([0]*20+[3]*20))
    max_error=0.; adam_error=0.
    for seed in range(5):
        w=initial_matrix(seed)
        d,f=Model('direct',w),Model('factorized',w)
        max_error=max(max_error,(d(torch.cat([x,xt]))-f(torch.cat([x,xt]))).abs().max().item())
        assert max_error<1e-6
        assert f.A.shape==(40,40) and f.B.shape==(82,40)
    for kind in ['direct','factorized']:
        a,b=Model(kind,initial_matrix(0)),Model(kind,initial_matrix(0))
        oa,ob=optimizer_for(a,'adam',.01,0),optimizer_for(b,'adamw',.01,0)
        for step in range(100):
            for m,o in [(a,oa),(b,ob)]:
                o.zero_grad(); torch.nn.functional.cross_entropy(m(x),y).backward(); o.step()
        err=(a.matrix()-b.matrix()).abs().max().item()
        adam_error=max(adam_error,err); assert err<1e-12
    w=initial_matrix(7).requires_grad_()
    assert torch.autograd.gradcheck(lambda v:torch.nn.functional.cross_entropy(x@v.T,y),(w,),fast_mode=True)
    f=Model('factorized',initial_matrix(7))
    assert torch.autograd.gradcheck(lambda a,b:torch.nn.functional.cross_entropy(x@(a@b.T).T,y),(f.A,f.B),fast_mode=True)
    logits=torch.randn(10,40,dtype=torch.float64); target=torch.arange(10)
    assert torch.allclose(margin(logits,target),margin(logits+3.4,target))
    result=dict(data_split='passed',matching_logits_max_error=max_error,adam_adamw_zero_decay_max_error=adam_error,gradient_checks='passed',margin_shift_invariance='passed')
    Path('results/checks.json').write_text(json.dumps(result,indent=2)); print(result)
if __name__=='__main__': main()
