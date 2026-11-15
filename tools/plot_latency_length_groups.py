"""Accepted latency distributions by actual input length; no model execution."""
import argparse,json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox,HPacker,TextArea,DrawingArea
from paper_figures.acl835_plot_style import setup_acl,detail_caption

METHODS=[('kvcloak_unfused','KV-Cloak'),('kvcloak_fused','Fused'),('veilkv_prefix','VeilKV'),('aead_no_sharing','No sharing')]
COLORS=['#A6CEE3','#D2D2D2','#BDB2FF','#B2DF8A']
BINS=[(32,64),(64,128),(128,256),(256,512),(512,768)]
def rows(folder):
 files=[folder/'records.jsonl'] if (folder/'records.jsonl').exists() else sorted(folder.glob('records*.jsonl.gz'))
 for p in files:
  opener=gzip.open if p.suffix=='.gz' else open
  with opener(p,'rt') as f:
   for line in f:
    x=json.loads(line)
    if x['status']=='measured':yield x

def smallcaps():
 return HPacker(children=[TextArea('V',textprops=dict(fontfamily='Arial',fontsize=6.7)),TextArea('EIL',textprops=dict(fontfamily='Arial',fontsize=5.36)),TextArea('KV',textprops=dict(fontfamily='Arial',fontsize=6.7))],pad=0,sep=0,align='baseline')
def draw(source,output,receipt):
 setup_acl('bar');fig,axes=plt.subplots(1,2,figsize=(7.,2.95));fig.subplots_adjust(left=.075,right=.99,bottom=.25,top=.85,wspace=.19)
 groups={};audit={}
 for lo,hi in BINS:
  jid=f'latency_llama31_8b_float32_b16_l{lo}_{hi}';r=list(rows(source/jid));assert len(r)==9000
  for op in ['protect','recover']:
   for key,_ in METHODS:
    data=[x for x in r if x['operation']==op and x['method']==key];assert len(data)==1000
    groups[(lo,op,key)]=np.array([x['seconds']*1000 for x in data])
    audit[f'{jid}|{key}|{op}']=dict(n=1000,distinct_inputs=len({x['source_sid'] for x in data}),p50_ms=float(np.median(groups[(lo,op,key)])),p95_ms=float(np.percentile(groups[(lo,op,key)],95)),p99_ms=float(np.percentile(groups[(lo,op,key)],99)),all_values_ms=groups[(lo,op,key)].tolist())
 for ax,op,cap in zip(axes,['protect','recover'],['(a) Protection','(b) Recovery']):
  for i,(key,name) in enumerate(METHODS):
   series=[groups[(lo,op,key)] for lo,_ in BINS];pos=np.arange(5)+(i-1.5)*.18
   ax.boxplot(series,positions=pos,widths=.155,patch_artist=True,manage_ticks=False,
       boxprops=dict(facecolor=COLORS[i],edgecolor='.35',linewidth=.48),medianprops=dict(color='red',linewidth=.8),
       whiskerprops=dict(color='.4',linewidth=.5),capprops=dict(color='.4',linewidth=.5),
       flierprops=dict(marker='o',markerfacecolor='none',markeredgecolor='.45',markersize=1.35,markeredgewidth=.35))
  top=max(float(max(groups[(lo,op,k)])) for lo,_ in BINS for k,_ in METHODS)*1.06
  ax.set(xlim=(-.55,4.55),ylim=(0,top),xticks=range(5),xticklabels=[f'{a}\N{EN DASH}{b-1}' for a,b in BINS],ylabel='Time per call (ms)',xlabel='Input length (tokens)')
  ax.set_facecolor((.917969,.917969,.949219));ax.set_axisbelow(True);ax.grid(axis='y',color='white',linewidth=.75)
  ax.tick_params(axis='both',length=0,labelsize=6.5,pad=3);ax.set_ylabel(ax.get_ylabel(),fontsize=7.0,weight='bold',labelpad=4);ax.set_xlabel(ax.get_xlabel(),fontsize=7.0,labelpad=4)
  for s in ax.spines.values():s.set(color='black',linewidth=.65)
  for x in [.5,1.5,2.5,3.5]:ax.axvline(x,color='white',linewidth=.6,zorder=0)
  detail_caption(fig,(ax.get_position().x0+ax.get_position().x1)/2,.055,cap)
 items=[]
 for i,(key,name) in enumerate(METHODS):
  mark=DrawingArea(11,7,0,0)
  from matplotlib.patches import Rectangle
  mark.add_artist(Rectangle((0,1),10,6,facecolor=COLORS[i],edgecolor='black',linewidth=.45))
  label=smallcaps() if key=='veilkv_prefix' else TextArea(name,textprops=dict(fontfamily='Arial',fontsize=6.7))
  items.append(HPacker(children=[mark,label],pad=0,sep=3,align='center'))
 legend=AnchoredOffsetbox(loc='upper center',child=HPacker(children=items,pad=0,sep=13,align='center'),pad=.3,borderpad=.2,frameon=True,bbox_to_anchor=(.53,.995),bbox_transform=fig.transFigure);legend.patch.set(facecolor='white',edgecolor='black',linewidth=.6);fig.add_artist(legend)
 output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output);fig.savefig(output.with_suffix('.png'),dpi=220)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();lb=legend.get_window_extent(renderer);assert all(not lb.overlaps(ax.get_window_extent(renderer)) for ax in axes)
 result=dict(primary_profile='Llama-3.1-8B FP32 KV-Cloak block16',length_groups=[list(x) for x in BINS],timed_calls_per_method_operation_length=1000,all_outliers_retained=True,y_axis='raw wall time per call in ms',source_style='Approved PROMPTPEEK boxplot geometry and background; ACL835 Figure4 Arial and lower serif subcaptions',adaptation='Five length groups with four method fills; legend above axes; native Arial smallcaps defense name',other_figures_unchanged=True,series=audit)
 receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(result,indent=2)+'\n');plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();draw(a.results,a.output,a.receipt)
