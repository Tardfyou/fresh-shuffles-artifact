"""Reflow the six accepted scaling settings into two stacked single-column panels."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_figures.acl835_plot_style import setup_acl,detail_axes,detail_caption,save_stat


def draw(points,out):
    setup_acl('bar');assert len(points)==6
    cohorts={p['sample_ids_improved']+p['sample_ids_unchanged']+p['sample_ids_degraded'] for p in points}
    assert len(cohorts)==1
    cohort=cohorts.pop()
    colors=dict(blue=(.12207,.470947,.705078),red=(.890625,.102051,.110107),better=(.650391,.808594,.890625),equal=(.697266,.875,.541016),worse=(.984375,.603516,.599609),background=(.917969,.917969,.949219))
    fig=plt.figure(figsize=(3.375,3.85))
    top=fig.add_axes([.31,.64,.66,.31]);bottom=fig.add_axes([.31,.16,.66,.31]);y=np.arange(6)
    gains=[p['gain_percentage_points'] for p in points]
    names=[('Qwen' if p['model'].startswith('Qwen') else 'Llama')+f' · {p["ratio"]}' for p in points]
    top.barh(y,gains,height=.56,color=[colors['blue'] if x>=0 else colors['red'] for x in gains],edgecolor='black',lw=.45,zorder=3)
    top.axvline(0,color='.3',lw=.55)
    for i,p in enumerate(points):
        top.text(-.055,i-.19,names[i],transform=top.get_yaxis_transform(),ha='right',va='center',fontsize=6.8)
        top.text(-.055,i+.31,f'{p["same_epoch"]["correct"]:,} → {p["adapted"]["correct"]:,}',transform=top.get_yaxis_transform(),ha='right',va='center',fontsize=5.6,color='.28')
        top.annotate(f'{p["net_head_token_gain"]:+,}',(max(0,gains[i]),i),xytext=(3,0),textcoords='offset points',ha='left',va='center',fontsize=6.4)
    xmax=max(1.20,np.ceil(max(gains)*1.18/.4)*.4)
    top.set(xlim=(-.04,xmax),xticks=np.arange(0,xmax+.01,.4),yticks=y,yticklabels=[],xlabel=r'Gain in $R_{\mathrm{bag}}$ (percentage points)')
    accum=np.zeros(6)
    for key,color,label in [('sample_ids_improved',colors['better'],'Better'),('sample_ids_unchanged',colors['equal'],'Equal'),('sample_ids_degraded',colors['worse'],'Worse')]:
        values=np.array([p[key] for p in points]);bottom.barh(y,values,left=accum,height=.56,color=color,edgecolor='black',lw=.4,label=label,zorder=3)
        for yy,start,value in zip(y,accum,values):
            if value>=8:bottom.text(start+value/2,yy,str(value),ha='center',va='center',fontsize=6.5)
            elif value:bottom.annotate(str(value),(start+value/2,yy),xytext=(0,-9),textcoords='offset points',ha='center',va='center',fontsize=6.)
        accum+=values
    assert np.all(accum==cohort)
    ticks=[0,25,50,75,100] if cohort==100 else [0,150,300,450,cohort]
    bottom.set(xlim=(0,cohort),xticks=ticks,yticks=y,yticklabels=names,xlabel=f'Paired prompts (of {cohort})')
    for ax in [top,bottom]:
        ax.set_ylim(5.58,-.58);detail_axes(ax);ax.set_facecolor(colors['background']);ax.grid(color='white',lw=.65,linestyle='-');ax.set_axisbelow(True)
        ax.tick_params(axis='x',labelsize=6.1);ax.tick_params(axis='y',labelsize=6.8,length=0,pad=4)
        ax.set_xlabel(ax.get_xlabel(),fontsize=6.8,labelpad=3)
        for s in ax.spines.values():s.set(color='black',lw=.6)
        ax.axhline(3.5,color='.62',lw=.5,ls=(0,(2,2)))
    bottom.legend(loc='lower center',bbox_to_anchor=(.5,1.025),ncol=3,frameon=False,fontsize=6.3,handlelength=1.,columnspacing=1.)
    detail_caption(fig,.64,.54,'(a) Extra hits and normalized gain')
    detail_caption(fig,.64,.045,'(b) Paired prompt outcomes')
    out.mkdir(parents=True,exist_ok=True)
    save_stat(fig,'scaling_followup','Approved scaling figure: unchanged data/palette, two vertically stacked panels at native single-column width',out)
    return dict(layout='two rows, one column',canvas_inches=[3.375,3.85],conditions=points,counts_and_denominators_unchanged=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();d=json.loads(a.data.read_text());result=draw(d['scaling'],a.output);a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(result,indent=2)+'\n');print(a.receipt)
