"""Dense statistical views of frozen observations; no recovery or model calls."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from paper_figures.acl835_plot_style import setup_acl, detail_axes, detail_caption, detail_legend, save_stat
from paper_figures.visual_palette import TEAL, CORAL

# Visible source palettes, measured from the actual security-paper PDFs.
# USENIX Security 2026, Approximate Caches, Figures 11 and 22.
SEC_PURPLE=(.6,.6,1.)
SEC_GREY=(.8,.8,.8)
SEC_RED=(1.,0.,0.)
SEC_BLUE=(0.,0.,1.)
# NDSS 2025, PROMPTPEEK, Figures 10/11 and 13.
PEEK_BOX=(.705078,.832031,.873047)
PEEK_BG=(.917969,.917969,.949219)
PEEK_LIGHT_BLUE=(.650391,.808594,.890625)
PEEK_BLUE=(.12207,.470947,.705078)
PEEK_LIGHT_GREEN=(.697266,.875,.541016)
PEEK_PINK=(.984375,.603516,.599609)
PEEK_RED=(.890625,.102051,.110107)


def finish(fig, name, output, source, values):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    audit = output.parent/'build'; audit.mkdir(parents=True, exist_ok=True)
    save_stat(fig, name, source, output)
    (audit/(name+'-visual-data.json')).write_text(json.dumps({'reference': source, 'plotted_values': values,
                                                          'font': 'Arial', 'subcaption_font': 'STIXGeneral'}, indent=2)+'\n')


def privacy(data, output):
    setup_acl('bar')
    kinds = ['ssn', 'credit', 'password', 'mrn', 'name', 'address']
    names = ['SSN', 'Card', 'Password', 'MRN', 'Name', 'Address']
    fig=plt.figure(figsize=(3.375,2.25));ax=fig.add_axes([.15,.23,.82,.64])
    expected=[100*data['privacy_summary']['by_type'][k]['top1'] for k in kinds]
    unique=[100*sum(r['unique_top1'] for r in data['privacy_cases'] if r['etype']==k)/data['privacy_summary']['by_type'][k]['n'] for k in kinds]
    x=np.arange(6)
    for offset,ys,color,label in [(-.17,expected,SEC_PURPLE,'Expected top-1'),(.17,unique,SEC_GREY,'Uniquely first')]:
        ax.bar(x+offset,ys,.34,color=color,edgecolor='black',linewidth=.55,label=label,zorder=3)
    ax.set(xticks=x,xticklabels=names,ylim=(0,112),yticks=[0,25,50,75,100],ylabel='Identification (%)')
    detail_axes(ax);ax.tick_params(axis='x',labelsize=5.8)
    for spine in ax.spines.values():spine.set(color='black',linewidth=.7)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.56,.995),ncol=2,frameon=False,fontsize=6)
    fig.text(.55,.075,'16–17 cases/type · 100 candidates/case',ha='center',fontsize=6)
    values={'kinds':kinds,'expected_top1_percent':expected,'uniquely_first_percent':unique}
    finish(fig,'privacy',output,'USENIX Security2026 Approximate Caches Figure22: compact paired bars, source lavender (.6,.6,1) and grey (.8,.8,.8).',values)


def scaling(data, output):
    setup_acl('bar')
    points = data['scaling']; y = np.arange(6)
    fig = plt.figure(figsize=(7, 2.95))
    left = fig.add_axes([.20, .19, .32, .70]); right = fig.add_axes([.61, .19, .37, .70])
    gains = [p['gain_percentage_points'] for p in points]
    names = [(('Qwen' if p['model'].startswith('Qwen') else 'Llama')+f' · {p["ratio"]}') for p in points]
    left.barh(y, gains, height=.57, color=[PEEK_BLUE if x >= 0 else PEEK_RED for x in gains], ec='black', lw=.45, zorder=3)
    left.axvline(0, color='.25', lw=.6)
    left.set(yticks=y, yticklabels=[], xlim=(-.09, 1.6), xlabel='Recovery gain (percentage points)')
    left.set_ylim(5.65,-.65); detail_axes(left)
    left.tick_params(axis='y', labelsize=7)
    for i, p in enumerate(points):
        left.annotate(f'{p["net_head_token_gain"]:+,}', (max(0,gains[i]), i), xytext=(4, 0),
                      textcoords='offset points', ha='left', va='center', fontsize=6)
        # Exact correct counts travel with the figure, rather than a duplicate table.
        left.text(-.035,i-.11,names[i],transform=left.get_yaxis_transform(),ha='right',va='center',fontsize=6.6)
        left.text(-.035,i+.18,f'{p["same_epoch"]["correct"]:,} → {p["adapted"]["correct"]:,}',
                  transform=left.get_yaxis_transform(),ha='right',va='center',fontsize=5.1,color='.28')
    accumulated = np.zeros(6)
    for key, color, label in [('sample_ids_improved', PEEK_LIGHT_BLUE, 'Better'), ('sample_ids_unchanged', PEEK_LIGHT_GREEN, 'Equal'), ('sample_ids_degraded', PEEK_PINK, 'Worse')]:
        values = np.array([p[key] for p in points])
        right.barh(y, values, left=accumulated, height=.57, color=color, ec='black', lw=.4, label=label, zorder=3)
        for yy, start, value in zip(y, accumulated, values):
            if value >= 8:
                right.text(start+value/2, yy, str(value), ha='center', va='center', fontsize=6, color='black')
            elif value > 0:
                # Keep tiny segments legible without moving their boundaries.
                right.annotate(str(value), (start+value/2, yy), xytext=(0, -10), textcoords='offset points',
                               ha='center', va='center', fontsize=5.6)
        accumulated += values
    assert np.all(accumulated == 100)
    right.set(yticks=y, yticklabels=[], xlim=(0,100), xticks=[0,25,50,75,100], xlabel='Paired prompts (of 100)')
    right.invert_yaxis(); detail_axes(right)
    for ax in [left, right]:
        ax.set_facecolor(PEEK_BG)
        ax.grid(color='white',linewidth=.7,linestyle='-')
        for spine in ax.spines.values():spine.set(color='black',linewidth=.65)
        ax.axhline(3.5, color='.65', lw=.6, ls=(0,(2,2)))
        ax.set_ylim(5.65,-.65)
    legend = right.legend(loc='lower center', bbox_to_anchor=(.5, 1.025), ncol=3, frameon=False,
                          fontsize=6, handlelength=1.0, columnspacing=.9)
    detail_caption(fig,.36,.045,'(a) Extra head–token hits and normalized gain')
    detail_caption(fig,.79,.045,'(b) Prompt outcomes at every setting')
    finish(fig, 'scaling_followup', output, 'NDSS2025 PROMPTPEEK Figure13: horizontal outcome stacks and original measured blue/green/pink palette; normalized to paired prompts.', points)


def budget(data, output):
    setup_acl('bar')
    rules = [('main','Main score'), ('gram','Gram spectrum'), ('sorted_row_norm','Sorted norms'),
             ('procrustes','Public alignment'), ('b1','Direct dictionary')]
    fig = plt.figure(figsize=(7, 2.7))
    axes = [fig.add_axes([.20,.21,.35,.66]), fig.add_axes([.61,.21,.35,.66])]
    stored = []
    for ax, model, caption in zip(axes, ['Qwen2.5-0.5B','Llama-3.2-1B'], ['(a) Qwen2.5-0.5B','(b) Llama-3.2-1B']):
        for index, (key, name) in enumerate(rules):
            for offset, scope, color, marker in [(-.17,'first',SEC_PURPLE,'o'),(.17,'all',SEC_GREY,'s')]:
                values = data['budget_comparison'][model]['hits_by_seed'][key+'_'+scope]
                pair = [values['seed5001'],values['seed5002']]; yy = index+offset
                ax.barh(yy,np.mean(pair),height=.31,color=color,edgecolor='black',lw=.55,zorder=2)
                ax.plot([min(pair),max(pair)],[yy,yy],color='black',lw=.9,zorder=3)
                x = max(pair)+3 if max(pair)<86 else min(pair)-3
                ax.text(x,yy,f'{pair[0]}/{pair[1]}',ha='left' if max(pair)<86 else 'right',va='center',fontsize=5.8)
                stored.append({'model':model,'rule':key,'heads':scope,'seed_hits':pair})
        ax.set(xlim=(0,105),xticks=[0,25,50,75,100],yticks=range(5),yticklabels=[r[1] for r in rules] if ax==axes[0] else [],xlabel='Mean correct candidates (of 100)')
        ax.set_ylim(4.6,-.6); detail_axes(ax);ax.tick_params(axis='y',labelsize=6.7)
        for spine in ax.spines.values():spine.set(color='black',linewidth=.7)
        detail_caption(fig,(ax.get_position().x0+ax.get_position().x1)/2,.055,caption)
    handles=[Patch(fc=SEC_PURPLE,ec='black',lw=.5,label='First head'),Patch(fc=SEC_GREY,ec='black',lw=.5,label='All heads')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.59,.995),ncol=2,frameon=False,fontsize=6.5)
    finish(fig,'candidate_budget',output,'USENIX Security2026 Approximate Caches Figure22: paired bars and original lavender/grey; thin ranges retain both seeds.',stored)


def ordering(data, output):
    setup_acl('bar')
    fig=plt.figure(figsize=(7,3.25))
    api=data['api_summary']['models'];ax=fig.add_axes([.10,.22,.22,.69]);stored={}
    metrics=[('macro_position_percent','Position'),('macro_multiset_recall_percent','Bag recall'),('macro_multiset_precision_percent','Bag precision')]
    for offset,model,color,label in [(-.17,'gpt-6.1-sol',SEC_PURPLE,'Sol'),(.17,'gpt-6-luna',SEC_GREY,'Luna')]:
        values=[api[model][key] for key,_ in metrics]
        ax.barh(np.arange(3)+offset,values,.31,color=color,edgecolor='black',lw=.55,label=label,zorder=3)
        for yy,value in zip(np.arange(3)+offset,values):ax.text(value+1.5,yy,f'{value:.2f}',ha='left',va='center',fontsize=5.5)
        stored[model]=values
    ax.set(xlim=(0,110),xticks=[0,25,50,75,100],yticks=range(3),yticklabels=[name for _,name in metrics],xlabel='Mean score (%)')
    ax.set_ylim(2.6,-.6);detail_axes(ax);ax.tick_params(axis='y',labelsize=6)
    for spine in ax.spines.values():spine.set(color='black',linewidth=.65)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,fontsize=6)
    arrays={key:np.array([r['conditions'][key] for r in data['semantic_local']]) for key in ['end2end','oracle_ms','no_cross_block','random_order']}
    arrays.update({model:np.array([r['embedding_cosine'] for r in data['api_scores'][model]]) for model in ['gpt-6.1-sol','gpt-6-luna']})
    panels=[('end2end','oracle_ms','E2E','Oracle','(b) Recovered vs. truth bags'),
            ('end2end','no_cross_block','Context','No context','(c) Cross-block context'),
            ('gpt-6.1-sol','gpt-6-luna','Sol','Luna','(d) GPT output similarity'),
            ('end2end','random_order','E2E','Random','(e) Language prior vs. random')]
    bins=np.linspace(0,1,41)
    peak=max(np.histogram(v,bins)[0].max()/len(v)*100 for v in arrays.values())
    ceiling=np.ceil(peak/10)*10+5
    histogram=[]
    for index,(first,second,name1,name2,caption) in enumerate(panels):
        x=.425 if index%2==0 else .735;y=.625 if index<2 else .22
        small=fig.add_axes([x,y,.235,.285])
        for key,name,color in [(first,name1,SEC_RED),(second,name2,SEC_BLUE)]:
            values=arrays[key];assert len(values)==602
            small.hist(values,bins=bins,weights=np.full(len(values),100/len(values)),color=color,alpha=.3,ec=color,lw=.35,
                       label=f'{name}: {np.mean(values):.3f}')
            small.axvline(np.mean(values),color=color,lw=.55,ls='--')
            counts,_=np.histogram(values,bins)
            histogram.append({'panel':index,'condition':key,'n':len(values),'counts':counts.tolist(),'bins':bins.tolist(),'mean':float(np.mean(values))})
        xmin=0 if index==2 else .4
        assert all(min(arrays[k])>=xmin for k in [first,second])
        small.set(xlim=(xmin,1),ylim=(0,ceiling),xticks=[0,.5,1] if xmin==0 else [.4,.6,.8,1])
        small.grid(axis='y',color='.88',linewidth=.5);small.set_axisbelow(True)
        small.tick_params(labelsize=5.2,length=2,width=.5,pad=1.4)
        for spine in small.spines.values():spine.set(color='black',linewidth=.65)
        if index%2==0:small.set_ylabel('Frequency (%)',fontsize=5.9,labelpad=2)
        if index>=2:small.set_xlabel('Cosine',fontsize=5.9,labelpad=2)
        legend=small.legend(loc='upper left',fontsize=4.8,frameon=False,handlelength=1.,handletextpad=.3,borderaxespad=.3)
        detail_caption(fig,x+.1175,.555 if index<2 else .065,caption)
    detail_caption(fig,.1975,.065,'(a) GPT ordering metrics')
    stored['histograms']=histogram
    finish(fig,'ordering_distributions',output,'USENIX Security2026 Approximate Caches Figures11/22: 2x2 frequency histograms with source red/blue at alpha .3 and lavender/grey paired bars.',stored)


def timing(data, output):
    setup_acl('bar')
    fig,axes=plt.subplots(1,2,figsize=(7,2.55))
    fig.subplots_adjust(left=.075,right=.985,bottom=.31,top=.96,wspace=.18)
    methods=[('kvcloak','KV-Cloak'),('kvcloak_fused','Fused'),('tee','VeilKV'),('nosharing_aead','No sharing')]
    stored={}
    for ax,operation,caption in zip(axes,['protect','recover'],['(a) Protection','(b) Recovery']):
        groups=[np.array(data['cache_timing'][operation+'_'+key]['samples_seconds_per_1000_tokens'])*1000 for key,_ in methods]
        assert all(len(v)==100 for v in groups)
        ax.set_facecolor(PEEK_BG);ax.set_axisbelow(True);ax.grid(color='white',linewidth=.75)
        ax.boxplot(groups,positions=range(4),widths=.55,patch_artist=True,
                   boxprops={'facecolor':PEEK_BOX,'edgecolor':'.5','linewidth':.7},
                   medianprops={'color':'red','linewidth':.9},whiskerprops={'color':'.5','linewidth':.7},
                   capprops={'color':'.5','linewidth':.7},flierprops={'marker':'o','markerfacecolor':'none','markeredgecolor':'.5','markersize':2,'markeredgewidth':.5})
        labels=[name+'\n'+f'{np.median(values):.1f}' for (_,name),values in zip(methods,groups)]
        ax.set(xticks=range(4),xticklabels=labels,xlim=(-.6,3.6),ylim=(0,max(max(v) for v in groups)*1.09),ylabel='Time (ms / 1,000 tokens)')
        ax.tick_params(length=0,labelsize=6,pad=3);ax.set_ylabel(ax.get_ylabel(),fontsize=7.02,weight='bold',labelpad=4)
        for spine in ax.spines.values():spine.set(color='black',linewidth=.65)
        for (key,_),values in zip(methods,groups):stored[operation+'_'+key]={'n':100,'median_ms':float(np.median(values)),'p25_ms':float(np.percentile(values,25)),'p75_ms':float(np.percentile(values,75)),'all_values_ms':values.tolist()}
        detail_caption(fig,(ax.get_position().x0+ax.get_position().x1)/2,.075,caption)
    finish(fig,'cache_latency_distribution',output,'NDSS2025 PROMPTPEEK Figures10/11: measured light-blue boxes, red medians, lavender-grey background and white grids; Tukey whiskers and all outliers.',stored)


def render(data, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    privacy(data,output);scaling(data,output);budget(data,output);ordering(data,output);timing(data,output)
    from paper_figures.result_panels import plot_epochs
    plot_epochs(data['epochs'],output)
    return {'figures':['privacy','scaling_followup','candidate_budget','ordering_distributions','cache_latency_distribution','key_epochs'],
            'raw_experiment_values_changed':False,'table_matrix_palette':[TEAL,CORAL],
            'statistical_references':['USENIX Security2026 Approximate Caches Fig11/22','NDSS2025 PROMPTPEEK Fig10/11/13']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(render(json.loads(args.data.read_text()),args.out)))
