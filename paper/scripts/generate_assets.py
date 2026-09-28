#!/usr/bin/env python3
"""Regenerate manuscript figures/tables from frozen records; no model calls."""
import json, math, hashlib
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, TAB = (ROOT/x for x in ('evidence','figures','tables'))
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
def read(name): return json.loads((DATA/name).read_text())
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix',
    'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':7,'axes.linewidth':.6,'lines.linewidth':1,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.pad_inches':.035})
C=['#1f77b4','#ff7f0e','#2ca02c']; GREY='#666666'
def finish(ax):
    ax.grid(True,ls=':',lw=.45,color='#b7b7b7',alpha=.75)
    ax.set_axisbelow(True);ax.tick_params(width=.6,length=3)
def save(fig,name):
    # Normalize legacy double escapes before mathtext parses labels.
    for axis in fig.axes:
        for artist in axis.texts:
            value = artist.get_text()
            while chr(92) * 2 in value:
                value = value.replace(chr(92) * 2, chr(92))
            value = value.replace(chr(92) + 'mathsf T', 'T')
            artist.set_text(value.replace(chr(92) + 'n', chr(10)))
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)

# Mechanism: original composition, deliberately stylized rather than empirical.
fig,ax=plt.subplots(figsize=(7.05,2.25));ax.set_xlim(0,10.4);ax.set_ylim(0,3.4);ax.axis('off')
def box(x,y,w,h,s,color='#eef4fa'):
    ax.add_patch(Rectangle((x,y),w,h,fc=color,ec='black',lw=.65))
    ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=8)
def arrow(x1,y1,x2,y2):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=.75,color='black'))
for x,title in [(0,'(a) Legal repeated-token request'),(3.55,'(b) A small observable orbit'),(7.15,'(c) Transfer within a key epoch')]:
    ax.text(x,3.23,title,fontsize=8.5,fontweight='bold',va='top')
box(.05,2.24,2.8,.54,'text → tokenizer → prefix + repeats')
for j in range(4):
    ax.add_patch(Rectangle((.18,1.04+j*.22),.65,.19,fc=C[0],ec='black',lw=.45))
ax.text(1.0,1.45,r'$V=\mathbf{1}v^{\mathsf T}$'+'\nall rows identical',va='center',fontsize=9)
ax.text(.12,.53,'Fresh row permutations leave V unchanged.',fontsize=7.7)
arrow(2.95,1.75,3.45,1.75)
box(3.55,2.25,3.05,.53,r'$C_j=u w^{\mathsf T}+s_j a^{\mathsf T}$','#edf4eb')
for j in range(4):
    for r in range(4):
        ax.add_patch(Rectangle((3.75+j*.69,1.03+r*.19),.51,.17,fc=C[1] if r==j else '#d4e7f5',ec='black',lw=.4))
ax.text(5.08,.72,r'$b!$ permutations → $b$ marker states',ha='center',fontsize=8)
arrow(6.73,1.75,7.1,1.75)
box(7.2,2.23,3.0,.56,'state differences → equivalent secrets','#fff0df')
box(7.2,1.32,3.0,.56,'new request cache → row demixing')
box(7.2,.39,3.0,.56,'public dictionary → token multiset','#edf4eb')
arrow(8.7,2.21,8.7,1.9);arrow(8.7,1.3,8.7,.98)
ax.text(.1,.08,'Attacker observes protected caches; server secrets remain unavailable.',fontsize=7.5,color=GREY)
save(fig,'overview')

# Probe budget: derive exact one-head occupancy with a finite-state DP.
curve=read('probecurve_05b.json')['probe_curve'];ns=[r['blocks'] for r in curve]
b=16;p=np.zeros(b+1);p[0]=1;exact={0:0.0}
for n in range(1,max(ns)+1):
    q=np.zeros(b+1)
    for k in range(b+1):
        q[k]+=p[k]*k/b
        if k<b:q[k+1]+=p[k]*(b-k)/b
    p=q;exact[n]=p[b]
fig,ax=plt.subplots(figsize=(3.4,2.32))
ax.plot(np.array(ns)*b,[100*r['P_full_recovery'] for r in curve],'-o',color=C[0],ms=3.7,mec='black',mew=.45,label='Both heads: recovery (10 trials)')
ax.plot(np.arange(1,max(ns)+1)*b,[100*exact[n] for n in range(1,max(ns)+1)],'--',color=C[1],label='One head: exact occupancy')
ax.set(xlabel='Repeated tokens (b = 16)',ylabel='Success probability (%)',ylim=(-3,103),xlim=(0,4200))
ax.set_xticks([0,1024,2048,3072,4096]);finish(ax);ax.legend(loc='lower right',frameon=True,edgecolor='black',fancybox=False)
save(fig,'probe_budget')

# Key epochs, displayed as recorded (no invented raw counts/intervals).
ind=read('independent_eval.json');fig,axs=plt.subplots(1,2,figsize=(7.0,2.18))
names={'qwen05':'Qwen2.5-0.5B','llama1b':'Llama-3.2-1B','phi3':'Phi-3-mini'}
for idx,(key,d) in enumerate(ind.items()):
    for ax,f in zip(axs,['cond_recovery_mean','ctrl_acc']):
        ax.plot(range(1,6),[100*r[f] for r in d['epochs']],marker=['o','s','D'][idx],ms=4,mec='black',mew=.4,color=C[idx],label=names[key])
axs[0].set(ylabel='Token multiset recovery (%)',ylim=(99.15,100.05),title='(a) Per-epoch recovery (zoomed axis)')
axs[1].set(ylabel='Wrong-secret recovery (%)',ylim=(-.005,.27),title='(b) Fresh-secret control')
for ax in axs: ax.set(xlabel='Secret epoch',xticks=range(1,6));finish(ax)
axs[0].legend(loc='lower left',frameon=True,edgecolor='black',fancybox=False)
fig.tight_layout(w_pad=2.3);save(fig,'key_epochs')

# Closed-set privacy values, recomputed from integer tie summaries.
priv=read('privacy_v3.json')['per_entity'];types=['ssn','credit','password','mrn','name','address']
def probability(r,k):return min(1.,max(0.,(k-r['strictly_higher'])/r['tie_group_size']))
fig,ax=plt.subplots(figsize=(3.4,2.3));x=np.arange(6);w=.36
ys=[100*np.mean([probability(r,1) for r in priv if r['type']==t]) for t in types]
ax.bar(x,ys,w,color=C[0],ec='black',lw=.55,label='Expected top-1 (random ties)')
ax.axhline(1,color=C[1],ls='--',lw=1,label='Uniform guess: 1%')
ax.set(xticks=x,xticklabels=['SSN','Card','Password','MRN','Name','Address'],ylabel='Expected identification (%)',ylim=(0,110))
ax.tick_params(axis='x',labelrotation=22)
for i,y in enumerate(ys):ax.text(i,y+2,f'{y:.1f}',ha='center',fontsize=7)
finish(ax);ax.legend(loc='upper left',bbox_to_anchor=(0,1.31),frameon=True,edgecolor='black',fancybox=False)
save(fig,'privacy')

# Main breadth table, raw counts only.
stats=read('STATS.json'); labels={'llama32_1b':'Llama-3.2-1B','olmo2_1b':'OLMo-2-1B','phi3_mini':'Phi-3-mini',
    'qwen25_05b':'Qwen2.5-0.5B','qwen25_05b_b8':'Qwen2.5-0.5B','qwen25_05b_b32':'Qwen2.5-0.5B',
    'qwen25_15b':'Qwen2.5-1.5B','qwen25_3b':'Qwen2.5-3B','qwen25_7b':'Qwen2.5-7B',
    'qwen3_06b':'Qwen3-0.6B','qwen3_17b':'Qwen3-1.7B','tinyllama':'TinyLlama-1.1B'}
rows=[]
for row in stats['per_config']:
    if row['correct'] is None:continue
    tag=row['tag'];d=read(tag+'.json');c,n=row['correct'],row['total'];cc,nn=row['control_raw']
    assert d['probe_raw_entry']
    rows.append(f"{labels[tag]} & {d['b']} & {d['kv_heads']} & {d['head_dim']} & {d['probe_tokens']:,} & {c:,}/{n:,} & {100*c/n:.2f} & {cc}/{nn} \\")
    rows[-1]+='\\'
text='\n'.join(rows)+'\n'
text = chr(10).join(r.rstrip(chr(92)) + chr(92) * 2 for r in rows) + chr(10)
(TAB/'breadth_rows.tex').write_text(text)
tot=stats['global'];assert tot==[263029,263904]
unique=sum(r['strictly_higher']==0 and r['tie_group_size']==1 for r in priv)
assert unique==21 and len(priv)==30
summary={'breadth_raw':tot,'breadth_percent':100*tot[0]/tot[1],
    'privacy_n':30,'privacy_unique_first':unique,'privacy_g_zero':sum(r['strictly_higher']==0 for r in priv),
    'privacy_expected':{str(k):sum(probability(r,k) for r in priv)/len(priv) for k in [1,5,10]},
    'key_epoch_minima':{k:min(e['cond_recovery_mean'] for e in d['epochs']) for k,d in ind.items()},
    'query_curve_trials_per_budget':10,'query_curve_empirical_scope':'two heads, fixed secret configuration',
    'query_curve_theory_scope':'one-head exact occupancy; newly calculated from b=16, no model run'}
(ROOT/'review/derived_numbers.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
