"""Plot six saved scaling observations; no inference or fitted-method execution."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from paper_figures.acl835_plot_style import (
    setup_acl, detail_axes, detail_caption, detail_legend, audit_details,
    save_stat, BAR_BLUE, BAR_PINK,
)


def plot(data, output):
    setup_acl('bar')
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.15), gridspec_kw={'width_ratios': [4, 2]})
    fig.subplots_adjust(left=.075, right=.985, top=.96, bottom=.25, wspace=.19)
    labels = []; bars = []; legends = []; values = []
    for index, (model, title) in enumerate([('Qwen2.5-0.5B', '(a) Qwen2.5-0.5B'), ('Llama-3.2-1B', '(b) Llama-3.2-1B')]):
        ax = axes[index]; points = [p for p in data['E4'] if p['model'] == model]
        x = np.arange(len(points)); width = .31
        for offset, condition, color, label in [(-width/2, 'same_epoch', BAR_BLUE, 'Unchanged decoder'), (width/2, 'adapted', BAR_PINK, 'Fitted variant')]:
            ys = [100*(p[condition]['total']-p[condition]['correct'])/p[condition]['total'] for p in points]
            container = ax.bar(x+offset, ys, width, color=color, edgecolor='black', linewidth=.4567, label=label, zorder=3)
            bars.extend(container)
            for bar, y in zip(container, ys):
                labels.append(ax.annotate(f'{y:.3f}', (bar.get_x()+bar.get_width()/2, y),
                                          xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=5.5))
            values.append({'model': model, 'condition': condition, 'miss_percent': ys})
        ax.set_xticks(x, [str(p['ratio']) for p in points])
        ax.set_xlim(-.6, len(points)-.4); ax.set_ylim(0, 1.72)
        ax.set_yticks([0, .5, 1, 1.5]); ax.set_xlabel('Scaling ratio', fontsize=7.02, weight='bold', labelpad=3)
        if index == 0:
            ax.set_ylabel('Miss rate (%)', labelpad=4)
            legends.append(detail_legend(ax, loc='upper left', bbox_to_anchor=(.025, .98)))
        else:
            ax.tick_params(labelleft=False)
        detail_axes(ax)
        detail_caption(fig, (ax.get_position().x0+ax.get_position().x1)/2, .045, title)
    audit_details(fig, 'scaling_followup', output, values, legends=legends, labels=labels, bars=bars)
    save_stat(fig, 'scaling_followup', 'ACL835 Figure 4a: paired pastel bars, zero baseline, common scale', output)
    return {'panels': 2, 'recorded_points': 6, 'zero_baseline': True, 'shared_y_limits_percent': [0, 1.72], 'source': 'Saved complete-block counts only.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(json.loads(args.data.read_text()), args.out)))
