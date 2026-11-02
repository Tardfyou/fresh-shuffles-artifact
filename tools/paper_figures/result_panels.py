"""Saved defensive latency panels only."""
import numpy as np
from .acl835_plot_style import setup_acl, save_stat, BAR_BLUE, BAR_PINK

def plot_latency(perf, out_dir):
    import matplotlib.pyplot as plt
    setup_acl('bar')
    fig = plt.figure(figsize=(3.375, 2.08))
    axes = [fig.add_axes([.13, .235, .35, .71]), fig.add_axes([.63, .235, .35, .71])]
    quantiles = ['p50', 'p95', 'p99']
    labels = []
    rectangles = []
    values_record = {}
    import math
    maxima=[max(1000*perf[op+'_'+name][q] for name in ['kvcloak','tee'] for q in quantiles) for op in ['protect','recover']]
    limits=[max(330,math.ceil(maxima[0]*1.35/50)*50),max(130,math.ceil(maxima[1]*1.15/20)*20)]
    ticks=[[0,100,200,300] if limits[0]<=350 else [0,100,200,300,400],
           [0,40,80,120] if limits[1]<=140 else [0,60,120,180]]
    for ax, operation, limit, ticks in zip(axes, ['protect', 'recover'], limits,ticks):
        for offset, name, color, label in zip([-.165, .165], ['kvcloak', 'tee'], [BAR_BLUE, BAR_PINK],
                                            ['KV-Cloak (unfused)', 'VeilKV TEE profile']):
            item = perf[operation+'_'+name]
            assert len(item['samples_seconds_per_1000_tokens']) == 100
            values = [1000*item[q] for q in quantiles]
            values_record[operation+'_'+name] = values
            xx = np.arange(3)+offset
            bars = ax.bar(xx, values, width=.33, color=color, edgecolor='black',
                          linewidth=.4567, label=label, zorder=3)
            rectangles.extend(bars)
            for x, value in zip(xx, values):
                # A small left offset keeps the lower bar's number clear of
                # the adjacent taller bar, while keeping it close to its top.
                labels.append(ax.annotate(f'{value:.1f}', (x+.165 if name=='kvcloak' else x, value), xytext=(-1.5 if name=='kvcloak' else 0, 2.2),
                                          textcoords='offset points', ha='right' if name=='kvcloak' else 'center', va='bottom',
                                          fontsize=5.515, color='black', zorder=4))
        ax.set(xlim=(-.58, 2.58), ylim=(0, limit), xticks=range(3), xticklabels=quantiles,
               yticks=ticks, ylabel='Time (ms)')
        ax.tick_params(axis='both', length=0, pad=2.2, labelsize=5.741)
        for label in ax.get_xticklabels():
            label.set_fontsize(7.02); label.set_fontweight('bold'); label.set_color('black')
        ax.set_ylabel('Time (ms)', fontsize=7.02, weight='bold', labelpad=3)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='.86', linewidth=.3654, linestyle=(0, (1.2, 2.2)), zorder=0)
        for separator in [.5, 1.5]: ax.axvline(separator, color='.82', linewidth=.4567, zorder=0)
        for side in ax.spines.values(): side.set_color('.8'); side.set_linewidth(.4567)
    # One boxed legend serves both panels; it sits above all left-panel bars.
    legend = axes[0].legend(loc='upper left', bbox_to_anchor=(.025, .987), fontsize=5.8,
                            frameon=True, fancybox=False, framealpha=1., facecolor='white',
                            edgecolor='black', borderpad=.32, labelspacing=.28, handlelength=1.25,
                            handletextpad=.45, borderaxespad=0.)
    legend.get_frame().set_linewidth(.65)
    for artist in legend.get_texts(): artist.set_color('black')
    for x, caption in [(.305, '(a) Protection'), (.805, '(b) Recovery')]:
        fig.text(x, .105, caption, ha='center', va='center', fontsize=8.1,
                 fontfamily='STIXGeneral', color='black')
    fig.text(.55, .025, 'Per 1,000 tokens', ha='center', va='center', fontsize=5.8, color='black')
    # Check the concrete placement of the internal legend and numeric labels.
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); legend_box=legend.get_frame().get_window_extent(renderer)
    for item in [*labels, *rectangles]:
        if legend_box.overlaps(item.get_window_extent(renderer)):
            raise ValueError(('latency legend overlaps data', getattr(item, 'get_text', lambda: 'bar')()))
    for label in labels:
        label_box=label.get_window_extent(renderer)
        for bar in rectangles:
            if label_box.overlaps(bar.get_window_extent(renderer)):
                raise ValueError(('latency numeric label overlaps a bar',label.get_text()))
    report = dict(canvas_bp=[243., 149.76], font='Arial', subcaption_font='STIXGeneral',
                  bar_edge_color='black', bar_edge_width_bp=.4567, axes_width_bp=.4567,
                  grid_width_bp=.3654, zero_baselines=True, top_and_bottom_aligned=True,
                  legend_inside=True, legend_overlaps_data=False, numeric_labels_overlap_bars=False,
                  values_ms_per_1000_tokens=values_record,
                  source='ACL835 Figure 4a; adapted to two operations and two methods',
                  differences=['Two rather than three bars per group', 'Separate panel ranges in milliseconds',
                               'STIXGeneral serif subcaptions instead of the source caption font'])
    from pathlib import Path
    import json
    audit_dir=Path(out_dir).parent/'build'
    audit_dir.mkdir(parents=True,exist_ok=True)
    (audit_dir/'latency-style-audit.json').write_text(json.dumps(report, indent=2))
    save_stat(fig, 'defense_latency', 'ACL835 Figure 4a: adjacent pastel bars, black contours, boxed internal legend, full grey axes, fine grids, Arial and lower serif panel captions', out_dir)
