#!/usr/bin/env python3
"""Generate ordering diagnostics from frozen records; no inference or network."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from audit_order_v2 import summarize
from f258_plot_style import F258, setup_f258, finish_f258, save_stat, STAT_LAYOUT

ROOT = Path(__file__).resolve().parents[1]
summary = summarize()
(ROOT / 'review/order_v2_audit.json').write_text(json.dumps(summary, indent=2) + '\n')

rows = [r'\setlength{\tabcolsep}{12bp}', r'\begin{tabular}{l|cccc}',
        r'\TableTop',
        r'\multirow{2}{*}{\textbf{Type}} & \multicolumn{4}{c}{\textbf{Nominal minimum blocks}} \\',
        r'\cline{2-5}', r' & \textbf{1} & \textbf{2} & \textbf{4} & \textbf{8} \\',
        r'\TableHeadRule']
for i, (category, label) in enumerate([('conversation', 'Conversation'),
                                      ('narrative', 'Narrative'),
                                      ('structured', 'Structured')]):
    cells = [c for c in summary['main']['nominal_cells'] if c['category'] == category]
    values = [100*c['conditions']['end2end']['macro_position_accuracy'] for c in cells]
    rows.append(label + ' & ' + ' & '.join(r'\RateCell{' + f'{v:.1f}' + '}' for v in values) + r' \\')
    if i < 2:
        rows.append(r'\hline')
rows.extend([r'\TableBottom', r'\end{tabular}', ''])
(ROOT / 'tables/order_v2_nominal.tex').write_text('\n'.join(rows))

setup_f258('line')
fig = plt.figure(figsize=(3.375, 2.40))
ax = fig.add_axes([.20, .23, .77, .57])
groups = [summary['main']['conditions'], summary['boundary']['random_seq']['conditions']]
x = np.arange(2)
series = [('end2end', 'End-to-end', F258['blue']),
          ('random_order', 'Random order', F258['orange']),
          ('no_cross_block', 'No cross-block', F258['green']),
          ('oracle_ms', 'Oracle bag', F258['black'])]
for i, (condition, label, color) in enumerate(series):
    values = [100*g[condition]['macro_position_accuracy'] for g in groups]
    positions = x + (i-1.5)*.17
    ax.bar(positions, values, .15, color=color, edgecolor='black',
           linewidth=.7237, label=label)
    for xx, value in zip(positions, values):
        ax.text(xx, value+.7, f'{value:.1f}', ha='center', va='bottom', fontsize=5.654)
ax.set(xlim=(-.6, 1.6), ylim=(0, 29), yticks=[0, 10, 20],
       xticks=x, xticklabels=['Main (600)', 'Random IDs (100)'],
       ylabel='Position accuracy (%)')
finish_f258(ax)
legend = fig.legend(*ax.get_legend_handles_labels(), loc='upper center',
    bbox_to_anchor=(.55, .995), ncol=2, fontsize=7.237, frameon=True,
    fancybox=False, framealpha=1, edgecolor='black', borderpad=.3,
    columnspacing=.8, handlelength=1.15, labelspacing=.25)
legend.get_frame().set_linewidth(.7237)
save_stat(fig, 'order_recovery', 'F258 measured font, bar edges, grid and legend', ROOT/'figures')
(ROOT/'review/order_v2_layout.json').write_text(json.dumps(STAT_LAYOUT, indent=2)+'\n')
print(json.dumps({'table': 'order_v2_nominal.tex', 'figure': 'order_recovery.pdf',
                  'main_records': summary['main']['records'],
                  'main_macro_pa': summary['main']['conditions']['end2end']['macro_position_accuracy']}, indent=2))
