#!/usr/bin/env python3
"""Check published plot data, frozen evidence, typography and visible table values."""
from pathlib import Path
from collections import Counter
import hashlib,json,re
import numpy as np
import pdfplumber
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'archive/20261012-before-f258-fig789'
read=lambda p:json.loads(p.read_text())
assert read(BASE/'review/derived_numbers.json')==read(ROOT/'review/derived_numbers.json')
for name,digest in read(BASE/'evidence-hashes.json').items():
    if name in ('evidence/manifest.json','evidence/FROZEN_CLAIMS.md'):
        continue # indexes/claim documentation extend as new result payloads arrive
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
layout=read(ROOT/'review/f258_stat_layout.json')
curve=read(ROOT/'evidence/probecurve_05b.json')['probe_curve']
assert layout['probe_budget']['line_data'][0][0]['xy']==[[r['blocks']*16,100*r['P_full_recovery']] for r in curve]
ind=read(ROOT/'evidence/independent_eval.json')
for panel,field in enumerate(['cond_recovery_mean','ctrl_acc']):
    for i,row in enumerate(ind.values()):
        assert layout['key_epochs']['line_data'][panel][i]['xy']==[[j+1,100*e[field]] for j,e in enumerate(row['epochs'])]
perf=read(ROOT/'evidence/veilkv_plus_v4.json')['perf']['data']
for panel,op in enumerate(['protect','recover']):
    for i,profile in enumerate(['kvcloak','tee']):
        assert layout['defense_latency']['line_data'][panel][i]['xy']==[[j,perf[f'{op}_{profile}'][q]] for j,q in enumerate(['p50','p95','p99'])]
priv=read(ROOT/'evidence/privacy_v3.json')['per_entity']
expected=[100*np.mean([min(1,max(0,(1-r['strictly_higher'])/r['tie_group_size'])) for r in priv if r['type']==kind]) for kind in ['ssn','credit','password','mrn','name','address']]
assert np.allclose(layout['privacy']['bar_heights'][0],expected,rtol=0,atol=1e-12)

def unstyle(text):
    for command,count,keep in [('RateCell',1,0),('RateShade',2,1),('multirow',3,2),('textbf',1,0)]:
        marker='\\'+command
        while marker in text:
            start=text.index(marker);pos=start+len(marker);args=[]
            for _ in range(count):
                assert text[pos]=='{',(command,text[pos:])
                depth=1;begin=pos+1;pos+=1
                while depth:
                    depth+=(text[pos]=='{')-(text[pos]=='}');pos+=1
                args.append(text[begin:pos-1])
            text=text[:start]+args[keep]+text[pos:]
    return text

def rows(path,columns,rate_index=None):
    result=[];previous=''
    for line in path.read_text().splitlines():
        if line.count('&')!=columns-1:continue
        line=unstyle(line).strip().rstrip('\\').strip();cells=[v.strip() for v in line.split('&')]
        if rate_index is not None and not re.fullmatch(r'[\d.]+',cells[rate_index]):continue
        if '\\' in line or 'Model' in line or 'Type' in line:continue
        if not any(re.search(r'\d',c) for c in cells[1:]):continue
        if not cells[0]:cells[0]=previous
        previous=cells[0];result.append(cells)
    return result
checks={}
for name,ncols,idx in [('app_epochs.tex',7,4),('app_diagnostics.tex',7,5),('app_privacy.tex',6,3)]:
    before=rows(BASE/'tables'/name,ncols,idx);after=rows(ROOT/'tables'/name,ncols,idx)
    assert before==after,(name,before,after);checks[name]=len(after)
before=rows(BASE/'sections/evaluation.tex',8,6);after=rows(ROOT/'sections/evaluation.tex',8,6)
assert before==after,(before,after);checks['breadth']=len(after)
fonts={};geometry={}
for name in layout:
    with pdfplumber.open(ROOT/'figures'/f'{name}.pdf') as d:
        pg=d.pages[0];names={c['fontname'].split('+')[-1] for c in pg.chars}
        assert all(n.startswith('TimesNewRomanPS') for n in names),(name,names)
        assert abs(pg.width-243)<.001
        assert layout[name]['all_text_inside_canvas']
        fonts[name]=sorted(names);geometry[name]=[pg.width,pg.height]
with pdfplumber.open(ROOT/'build/main.pdf') as d:
    text_colors=Counter(tuple(c['non_stroking_color']) if isinstance(c.get('non_stroking_color'),(tuple,list)) else c.get('non_stroking_color') for page in d.pages for c in page.chars)
    assert text_colors[(.463,.322,.604)]>20 # citation/URL purple after driver rounding
    assert text_colors[(.608,.227,.333)]>20 # cross-reference wine after rounding
out={'status':'pass','plotted_values_unchanged':True,'frozen_evidence_unchanged':True,
     'visible_table_rows_unchanged':checks,'fonts':fonts,'canvas_bp':geometry,
     'citation_rgb_output':[.463,.322,.604],'reference_rgb_output':[.608,.227,.333],
     'reference':'NDSS F258 Figures 7/8/9 and Table II',
     'deviations':['Plot types and data ranges follow our evidence; no synthetic densities or error bars.',
       'Legends moved where necessary to avoid data occlusion.',
       'Table widths adapt to different content; output RGB is rounded to 3 decimals by Tectonic.',
       'Citation purple and cross-reference wine are user-requested alternatives.']}
(ROOT/'review/f258_visual_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
