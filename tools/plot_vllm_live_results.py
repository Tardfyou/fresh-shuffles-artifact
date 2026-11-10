"""Plot audited real-engine cases; counts are read from every raw record."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from paper_figures.acl835_plot_style import setup_acl


def rows(path):
    plain=path/'records.jsonl'
    for file in ([plain] if plain.exists() else sorted(path.glob('records*.jsonl.gz'))):
        with (gzip.open(file,'rt') if file.suffix=='.gz' else file.open()) as f:
            for line in f:yield json.loads(line)


def extract(root):
    cells=[]
    for tag,model in [('qwen25_05b','Qwen0.5B'),('llama32_1b','Llama1B')]:
        for block in [16,32]:
            group=list(rows(root/f'defense_{tag}_b{block}'));assert len(group)==170
            counts=Counter(r['kind'] for r in group)
            assert counts==dict(functional=100,shared=20,isolation=10,tamper=40)
            for r in group:
                if r['kind']=='tamper':assert r['rejected_before_model_execution']
                else:assert r['functional_pass'] and r['comparison']['logits_exact'] and r['comparison']['tokens_equal']
            cells.append(dict(model=model,block=block,counts=dict(counts),functional_exact=130,tamper_rejected=40))
    attack=list(rows(root/'attack_qwen25_05b_b16'));assert len(attack)==100
    correct=sum(r['correct'] for r in attack);total=sum(r['total'] for r in attack)
    agreement=sum(r['known_key_consumer_comparison']['tokens_equal'] for r in attack)
    assert correct==total==41344 and agreement==90
    return dict(cells=cells,defense_functional_exact=520,defense_rejected=160,attack_correct=correct,attack_total=total,adapter_outputs_equal=agreement,adapter_outputs_total=100,formal_cases=780,smoke_not_included=50)


def draw(data,out):
    setup_acl('bar')
    # Count tiles adapt PROMPTPEEK Fig13's categorical stacks. Right-side
    # paired strips adapt Approximate Caches Fig22's lavender/grey outcomes.
    palette=dict(functional='#A6CEE3',shared='#B2DF8A',isolation='#FDBF6F',tamper='#CAB2D6')
    fig=plt.figure(figsize=(7,2.30),facecolor='white')
    ax=fig.add_axes([.175,.23,.435,.60]);right=fig.add_axes([.745,.30,.245,.53])
    title=dict(fontsize=8.2,fontweight='bold',color='#222222')
    fig.text(.34,.945,'VeilKV: exact bytes and logits',ha='center',**title)
    fig.text(.865,.945,'KV-Cloak: two outcomes',ha='center',**title)
    keys=['functional','shared','isolation','tamper'];names=['Ordinary: 100','Shared: 20','Isolation: 10','Tamper: 40']
    annotations=[];tile_count=0
    for i,cell in enumerate(data['cells']):
        x=0
        for key in keys:
            count=cell['counts'][key];assert count%10==0
            if key=='tamper':x+=.65
            for j in range(count//10):
                ax.add_patch(Rectangle((x+j-.38,i-.24),.76,.48,facecolor=palette[key],edgecolor='#364152',linewidth=.45));tile_count+=1
            x+=count//10
        label=f'{cell["model"]} · b={cell["block"]}'
        ax.text(-1.,i-.075,label,ha='right',va='center',fontsize=7.25,color='#18212D')
        ax.text(-1.,i+.17,'130/130 | 40/40',ha='right',va='center',fontsize=6.8,color='#4D5968')
    ax.set(xlim=(-.6,17.3),ylim=(3.58,-.58));ax.axis('off')
    # Break between different model families; group separators do not encode effects.
    ax.plot([-.4,17.1],[1.5,1.5],color='#B7C1CF',lw=.5,ls=(0,(2,2)))
    handles=[Patch(fc=palette[k],ec='#364152',lw=.4,label=n) for k,n in zip(keys,names)]
    legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.345,.888),ncol=4,frameon=False,fontsize=6.8,handlelength=1.1,columnspacing=1.05,handletextpad=.35)
    fig.text(.34,.165,'520/520 exact · 160/160 rejected',ha='center',fontsize=8.,weight='bold',color='#26384A')
    fig.text(.34,.065,'(a) Four engine cells; one tile represents 10 cases',ha='center',fontsize=7.3,fontfamily='STIXGeneral')
    fig.lines.append(plt.Line2D([.655,.655],[.16,.86],transform=fig.transFigure,color='#CFD5DE',lw=.65))
    bars=[dict(y=.28,label='Token-multiset recovery',percent=100.,count='41,344/41,344 head–tokens',color='#9999FF'),dict(y=1.13,label='Adapter output agreement',percent=90.,count='90/100 prompts',color='#CCCCCC')]
    for row in bars:
        y=row['y'];right.barh(y,100,height=.19,color='#F3F3F3',edgecolor='#303846',lw=.55,zorder=1)
        right.barh(y,row['percent'],height=.19,color=row['color'],edgecolor='#303846',lw=.55,zorder=2)
        if row['percent']<100:right.barh(y,100-row['percent'],left=row['percent'],height=.19,color='white',edgecolor='#7A8591',hatch='////',lw=.4,zorder=3)
        right.text(0,y-.20,row['label'],fontsize=7.25,ha='left',va='bottom',color='#222222')
        right.text(row['percent']-2,y,f'{row["percent"]:.0f}%',fontsize=7.2,ha='right',va='center',fontweight='bold',color='#202A37')
        right.text(0,y+.20,row['count'],fontsize=6.9,ha='left',va='top',color='#364152')
    right.set(xlim=(0,101),ylim=(1.62,-.42),xticks=[0,50,100],yticks=[])
    right.tick_params(axis='x',labelsize=6.8,length=2,width=.45,pad=1.5)
    for side in ['left','right','top']:right.spines[side].set_visible(False)
    right.spines['bottom'].set_color('#84909D');right.spines['bottom'].set_linewidth(.45)
    fig.text(.865,.165,'Qwen0.5B · b=16 · 100 prompts',ha='center',fontsize=6.9,color='#364152')
    fig.text(.865,.065,'(b) Separate metrics and denominators',ha='center',fontsize=7.3,fontfamily='STIXGeneral')
    out.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    outside=[]
    for text in fig.findobj(match=matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():continue
        bbox=text.get_window_extent(renderer)
        if bbox.x0<-.5 or bbox.x1>fig.bbox.width+.5 or bbox.y0<-.5 or bbox.y1>fig.bbox.height+.5:outside.append(text.get_text())
    assert not outside,outside
    fig.savefig(out/'vllm_live_results.pdf',metadata={'Author':'','Title':'End-to-end vLLM experiment results','Creator':'Matplotlib'})
    fig.savefig(out/'vllm_live_results.svg',metadata={'Creator':'Matplotlib'})
    fig.savefig(out/'vllm_live_results.png',dpi=220)
    plt.close(fig)
    return dict(tile_count=tile_count,all_numeric_labels_from_saved_records=True,vector_pdf=True,width_inches=7,height_inches=2.30,minimum_label_pt=6.8,out_of_canvas_text=outside,style='Adapted-readable: PROMPTPEEK Fig13 categorical count stacks + Approximate Caches Fig22 lavender/grey outcome comparison; Arial and STIX subcaptions',references=['NDSS2025 PROMPTPEEK, Fig13 (physical p12)','USENIX Security2026 Approximate Caches, Fig22 (physical p15)'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--data-output',type=Path,required=True);a=p.parse_args();data=extract(a.results);audit=draw(data,a.output);a.data_output.parent.mkdir(parents=True,exist_ok=True);a.data_output.write_text(json.dumps(dict(data=data,visual_validation=audit),indent=2)+'\n');print(json.dumps(audit,indent=2))
