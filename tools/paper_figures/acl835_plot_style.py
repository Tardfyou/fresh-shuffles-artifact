"""ACL-long.835 Figure 4 statistical styling at the final IEEE column width."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

BAR_GREEN=(.7019607843,1.,.6980392157)
BAR_BLUE=(.6980392157,.9450980392,1.)
BAR_PINK=(1.,.6980392157,.8235294118)
LINE_GOLD=(.9882352941,.8431372549,0.)
LINE_PINK=(1.,.6784313725,.6784313725)
LINE_PURPLE=(.7411764706,.6980392157,1.)
LINE_GREEN=(.6470588235,.9215686275,.3921568627)
STAT_LAYOUT={}


def setup_acl(profile='facet'):
    for path in (Path.home()/'.local/share/fonts/microsoft-core-fonts-arial').glob('*'):
        if path.suffix.lower()=='.ttf':font_manager.fontManager.addfont(str(path))
    font_manager.findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':6.89,'axes.labelsize':9.186 if profile=='facet' else 7.02,
                         'axes.labelweight':'bold','xtick.labelsize':9.186 if profile=='facet' else 7.02,'ytick.labelsize':5.741 if profile=='facet' else 5.014,
                         'legend.fontsize':6.89 if profile=='facet' else 6.518,'lines.linewidth':1.2631,
                         'axes.linewidth':.5741 if profile=='facet' else .4567,
                         'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavusans'})


def finish_acl(ax,profile='facet'):
    spine,grid=(.5741,.2871) if profile=='facet' else (.4567,.3654)
    dash=(1.85,.8) if profile=='facet' else (2.96,1.28)
    ax.set_axisbelow(True);ax.grid(True,color='.8',linestyle=(0,tuple(v/grid for v in dash)),linewidth=grid)
    ax.tick_params(width=grid,length=1.8,pad=2)
    for label in ax.get_xticklabels():label.set_fontweight('bold');label.set_color('.15')
    for label in ax.get_yticklabels():label.set_color('.15')
    for side in ax.spines.values():side.set(color='.8',linewidth=spine)


def legend_acl(ax,**kwargs):
    legend=ax.legend(frameon=True,fancybox=False,framealpha=1,facecolor='white',edgecolor='black',
                     borderpad=.25,labelspacing=.25,handlelength=1.5,**kwargs)
    legend.get_frame().set_linewidth(.4567)
    return legend


def detail_axes(ax, facet=False, separators=()):
    """The shared Figure 4 details, at the actual final column width."""
    spine,grid=(.5741,.2871) if facet else (.4567,.3654)
    ax.set_axisbelow(True)
    ax.grid(axis='both' if facet else 'y',color='.86',linewidth=grid,
            linestyle=(0,(1.2,2.2)),zorder=0)
    for x in separators:ax.axvline(x,color='.82',linewidth=spine,zorder=0)
    ax.tick_params(axis='both',length=0,pad=2.2,labelsize=5.741)
    ax.set_ylabel(ax.get_ylabel(),fontsize=7.02,weight='bold',labelpad=4)
    for label in ax.get_xticklabels():
        label.set_fontsize(7.02);label.set_fontweight('bold');label.set_color('black')
    for label in ax.get_yticklabels():label.set_color('black')
    for side in ax.spines.values():side.set_color('.8');side.set_linewidth(spine)


def detail_legend(ax, *args, **kwargs):
    options=dict(frameon=True,fancybox=False,framealpha=1.,facecolor='white',edgecolor='black',
                 fontsize=5.8,borderpad=.32,labelspacing=.28,handlelength=1.25,
                 handletextpad=.45,borderaxespad=0.)
    options.update(kwargs)
    legend=ax.legend(*args,**options);legend.get_frame().set_linewidth(.65)
    for item in legend.get_texts():item.set_color('black')
    return legend


def detail_caption(fig, x, y, text):
    return fig.text(x,y,text,ha='center',va='center',fontsize=8.1,
                    fontfamily='STIXGeneral',color='black')


def audit_details(fig, name, out_dir, data, legends=(), labels=(), bars=(), paths=()):
    """Check actual rendered labels and legend placement; retain plotted values."""
    import json
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for legend in legends:
        frame=legend.get_frame().get_window_extent(renderer)
        for item in [*labels,*bars]:
            if frame.overlaps(item.get_window_extent(renderer)):
                raise ValueError((name,'legend overlaps data or label'))
        for line in paths:
            points=line.get_transform().transform(line.get_xydata())
            for x,y in points:
                if frame.contains(x,y):raise ValueError((name,'legend covers a curve or error bar'))
    for item in labels:
        box=item.get_window_extent(renderer)
        for bar in bars:
            if box.overlaps(bar.get_window_extent(renderer)):
                raise ValueError((name,'numeric label intersects a bar'))
    audit_dir=Path(out_dir).parent/'build';audit_dir.mkdir(parents=True,exist_ok=True)
    report=dict(source='ACL835 Figure 4',font='Arial',subcaption_font='STIXGeneral',
                fixed_canvas_bp=[v*72 for v in fig.get_size_inches()],
                legend_overlaps_data=False,numeric_labels_overlap_bars=False,plotted_values=data)
    (audit_dir/(name+'-style-audit.json')).write_text(json.dumps(report,indent=2))


def save_stat(fig,name,reference,out_dir):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.get_size_inches()
    hidden=set()
    for ax in fig.axes:
        for axis in (ax.xaxis,ax.yaxis):
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc()<lo or tick.get_loc()>hi:hidden.update((tick.label1,tick.label2))
    outside=[]
    for artist in fig.findobj():
        if isinstance(artist,matplotlib.text.Text) and artist.get_visible() and artist.get_text() and artist not in hidden:
            b=artist.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
            if b.x0<-.002 or b.y0<-.002 or b.x1>w+.002 or b.y1>h+.002:outside.append(artist.get_text())
    if outside:raise ValueError((name,'text outside fixed canvas',outside))
    STAT_LAYOUT[name]=dict(reference=reference,canvas_bp=[w*72,h*72],font='Arial',
                           source_component='ACL-long.835 Figure 4',all_text_inside_canvas=True,
                           layout_adaptation='Own data and panel count; source statistical colors, type, strokes, and facet alignment')
    fig.savefig(Path(out_dir)/(name+'.pdf'),bbox_inches=None,pad_inches=0)
    fig.savefig(Path(out_dir)/(name+'.png'),dpi=240,bbox_inches=None,pad_inches=0)
    plt.close(fig)
