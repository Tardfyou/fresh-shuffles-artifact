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
C=['#2878a8','#df8230','#45864b']; GREY='#59636b'
def finish(ax):
    ax.grid(True,ls=':',lw=.4,color='#b7b7b7',alpha=.6)
    ax.set_axisbelow(True);ax.tick_params(width=.6,length=3)
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)

# Mechanism: original composition, deliberately stylized rather than empirical.
fig=plt.figure(figsize=(7.05,1.85))
ax=fig.add_axes([0,0,1,1])
ax.set(xlim=(0,7.05),ylim=(0,2.05));ax.axis('off')
box_labels=[]
def box(x,y,w,h,s,color='#eef4fa',fontsize=7.2):
    patch=Rectangle((x,y),w,h,fc=color,ec='#303b43',lw=.6)
    ax.add_patch(patch)
    label=ax.text(x+w/2,y+h/2,s,ha='center',va='center',
                  fontsize=fontsize,linespacing=1.3)
    box_labels.append((patch,label))
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',
                                mutation_scale=8,lw=.75,color='#303b43'))
for x,title,color in [(0.06,'(a) Repeated-token input',C[0]),
                      (2.45,'(b) Observable marker orbit',C[1]),
                      (4.84,'(c) Transfer within a key epoch',C[2])]:
    ax.text(x+1.075,1.94,title,fontsize=7.6,fontweight='bold',
            ha='center',va='center')
    ax.plot([x,x+2.15],[1.80,1.80],color=color,lw=1.2)
box(.06,1.28,2.15,.42,'Raw text → tokenizer\nprefix + repeated tokens')
for j in range(4):
    ax.add_patch(Rectangle((.28,.63+j*.105),.45,.087,
                           fc=C[0],ec='#303b43',lw=.45))
ax.text(1.42,.91,r'$V=\mathbf{1}v^{\mathsf{T}}$',
        ha='center',va='center',fontsize=9)
ax.text(1.42,.65,'identical Value rows',ha='center',fontsize=7)
ax.text(1.135,.28,'Row shuffling leaves\nrepeated Value rows unchanged.',
        ha='center',va='center',fontsize=7,linespacing=1.3,color=GREY)
arrow(2.25,1.49,2.42,1.49)
box(2.45,1.28,2.15,.42,r'$C_j=u w^{\mathsf{T}}+s_j a^{\mathsf{T}}$',
    '#fdf2e5',fontsize=8.6)
for j in range(4):
    for r in range(4):
        ax.add_patch(Rectangle((2.70+j*.46,.63+r*.105),.31,.087,
                               fc=C[1] if r==j else '#dceaf3',
                               ec='#303b43',lw=.4))
ax.text(3.525,.47,'Marker positions before row mixing',
        ha='center',fontsize=6.6,color=GREY)
ax.text(3.525,.24,r'$b!$ permutations → $b$ marker states',
        ha='center',fontsize=7.4)
arrow(4.64,1.49,4.81,1.49)
box(4.84,1.28,2.15,.42,'State differences\nestimate equivalent transforms','#edf4ec')
box(4.84,.72,2.15,.36,'Target cache → row demixing')
box(4.84,.15,2.15,.36,'Public dictionary → token multiset','#edf4ec')
arrow(5.915,1.26,5.915,1.10)
arrow(5.915,.70,5.915,.53)
# Measure labels in the exported coordinate system, including mathtext.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
layout=[]
for patch,label in box_labels:
    inner=patch.get_window_extent(renderer)
    text_box=label.get_window_extent(renderer)
    assert inner.x0+4 <= text_box.x0 and text_box.x1 <= inner.x1-4, label.get_text()
    assert inner.y0+3 <= text_box.y0 and text_box.y1 <= inner.y1-3, label.get_text()
    layout.append({'text':label.get_text(),'font_pt':label.get_fontsize(),
                   'horizontal_padding_px':min(text_box.x0-inner.x0,inner.x1-text_box.x1),
                   'vertical_padding_px':min(text_box.y0-inner.y0,inner.y1-text_box.y1)})
(ROOT/'review/figure_layout.json').write_text(
    json.dumps({'overview_box_labels':layout,'all_labels_fit':True},indent=2)+'\n')
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
ind=read('independent_eval.json');fig,axs=plt.subplots(1,2,figsize=(7.0,1.98))
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
ax.bar(x,ys,.48,color=C[0],ec='#303b43',lw=.55,label='Expected top-1 (random ties)')
ax.axhline(1,color=C[1],ls='--',lw=1,label='Uniform guess: 1%')
ax.set(xticks=x,xticklabels=['SSN','Card','Password','MRN','Name','Address'],ylabel='Expected identification (%)',ylim=(0,110))
ax.tick_params(axis='x',labelrotation=22)
for i,y in enumerate(ys):ax.text(i,y+2,f'{y:.1f}',ha='center',fontsize=7)
finish(ax);ax.legend(loc='upper left',bbox_to_anchor=(0,1.23),frameon=False)
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
