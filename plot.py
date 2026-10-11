"""Regenerate three figures and tidy tables exclusively from saved measurements."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'gd':'#176B87','adam':'#DC6B32','adamw0.01':'#8174A0','adamw0.03':'#32816B'}
LABELS={'gd':'GD','adam':'Adam','adamw0.01':'AdamW, decay 0.01','adamw0.03':'AdamW, decay 0.03'}

def condition(r):
    c=r['config']; return c['optimizer'] if c['optimizer']!='adamw' else 'adamw'+str(c['wd'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results');p.add_argument('--figures',default='figures');a=p.parse_args()
    root=Path(a.results); out=Path(a.figures);out.mkdir(exist_ok=True,parents=True)
    runs=[json.loads(f.read_text()) for f in sorted((root/'runs').glob('*.json'))]
    runs=[r for r in runs if condition(r)!='adamw0.0']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':220})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.7),sharey=True)
    for ax,kind in zip(axes,['direct','factorized']):
        for key in COLORS:
            subset=[r for r in runs if r['config']['kind']==kind and condition(r)==key]
            for j,r in enumerate(subset):
                h=r['history']
                ax.plot([v['train_ce'] for v in h],[100*v['test_accuracy'] for v in h],color=COLORS[key],alpha=.55,lw=1,label=LABELS[key] if j==0 else None)
            for t in [.1,.03,.01]:
                vals=[r['crossings'][str(t)]['test_accuracy']*100 for r in subset if str(t) in r['crossings']]
                if vals: ax.scatter([t],[np.mean(vals)],color=COLORS[key],s=22,zorder=4)
        for t in [.1,.03,.01]: ax.axvline(t,color='#BBBBBB',lw=.7,ls=':',zorder=0)
        ax.set_xscale('log');ax.set_xlim(4,1e-5);ax.set_ylim(-4,104)
        ax.set_title(kind.capitalize());ax.set_xlabel('Training cross-entropy (decreasing to the right)')
    axes[0].set_ylabel('Hidden-implication accuracy (%)');axes[1].legend(fontsize=8,loc='lower right')
    fig.suptitle('OCR at comparable training fit',fontweight='bold');fig.tight_layout();save(fig,'01_accuracy_vs_training_loss')
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.8),sharey=True)
    for ax,checkpoint,title in zip(axes,['0.01','final'],['First training CE <= 0.01','Fixed budget: 20,000 updates']):
        for i,key in enumerate(COLORS):
            for k,kind in enumerate(['direct','factorized']):
                sub=[r for r in runs if condition(r)==key and r['config']['kind']==kind]
                vals=[100*(r['final'] if checkpoint=='final' else r['crossings'][checkpoint])['test_accuracy'] for r in sub if checkpoint=='final' or checkpoint in r['crossings']]
                center=i+(-.17 if k==0 else .17)
                ax.scatter(center+np.linspace(-.065,.065,len(vals)),vals,color=COLORS[key],marker='o' if k==0 else '^',s=35,edgecolors='white',linewidth=.4,zorder=3)
                if vals: ax.plot([center-.1,center+.1],[np.mean(vals)]*2,color='black',lw=1.5)
        ax.set_xticks(range(4),['GD','Adam','AdamW\n0.01','AdamW\n0.03']);ax.set_title(title);ax.set_ylim(-4,104)
    axes[0].set_ylabel('Hidden-implication accuracy (%)')
    axes[1].scatter([],[],marker='o',color='#555555',label='Direct');axes[1].scatter([],[],marker='^',color='#555555',label='Factorized');axes[1].legend(loc='center right',fontsize=9)
    fig.suptitle('Five paired seeds per condition; black line = mean',fontweight='bold');fig.tight_layout();save(fig,'02_parameterization_optimizer')
    fig,ax=plt.subplots(figsize=(7.5,4.2))
    for key in COLORS:
        for kind,marker in [('direct','o'),('factorized','^')]:
            sub=[r['crossings']['0.01'] for r in runs if condition(r)==key and r['config']['kind']==kind and '0.01' in r['crossings']]
            ax.scatter([r['spectral_ratio'] for r in sub],[100*r['test_accuracy'] for r in sub],c=COLORS[key],marker=marker,s=48,alpha=.7,label=LABELS[key]+' / '+kind)
    ax.set_xlabel(r'Centered spectral ratio $\|W_c\|_* / \|W_c\|_F$');ax.set_ylabel('Hidden-implication accuracy (%)');ax.set_ylim(-4,104)
    ax.set_title('Spectrum and OCR at first training CE <= 0.01',fontweight='bold');ax.legend(fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left')
    fig.tight_layout();save(fig,'03_spectrum_vs_accuracy')
    rows=[]
    for r in runs:
        for checkpoint in ['0.1','0.03','0.01','final']:
            m=r['final'] if checkpoint=='final' else r['crossings'].get(checkpoint)
            rows.append(dict(**r['config'],checkpoint=checkpoint,reached=m is not None,**(m or {})))
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (root/'summary.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,keys);wr.writeheader();wr.writerows(rows)
    groups=[]
    for key in COLORS:
        for kind in ['direct','factorized']:
            sub=[r for r in runs if condition(r)==key and r['config']['kind']==kind]
            for cp in ['0.1','0.03','0.01','final']:
                ms=[r['final'] if cp=='final' else r['crossings'].get(cp) for r in sub];ms=[m for m in ms if m]
                g=dict(condition=key,kind=kind,checkpoint=cp,reached=len(ms))
                for metric in ['test_accuracy','train_ce','spectral_ratio','test_margin_mean','test_margin_min','normalized_nuclear']:
                    v=[m[metric] for m in ms];g[metric+'_mean']=float(np.mean(v)) if v else None
                    g[metric+'_min']=float(np.min(v)) if v else None;g[metric+'_max']=float(np.max(v)) if v else None
                groups.append(g)
    (root/'aggregates.json').write_text(json.dumps(groups,indent=2))
    print('Generated three figures, summary.csv, and aggregates.json')
if __name__=='__main__': main()
