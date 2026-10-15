"""F258 Figure 7--9 typography and plot primitives, measured in PDF bp.

Requires NumPy/Matplotlib and a locally licensed Times New Roman installation.
Use fixed canvases at their final embedding width. Scientific data are supplied
by the caller; this module does not create or change experiment results.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Source-measured NDSS F258 Figures 7--9. Statistical assets use fixed canvas
# widths matching IEEEtran's actual 243 bp column / 504 bp full text width.
# Microsoft Core Fonts are installed locally, never copied from the source PDF.
F258 = {'blue':'#1f77b4', 'orange':'#ff7f0e', 'green':'#2ca02c',
        'red':'#d62728', 'black':'#1a1a1a',
        'threshold_red':'#ff0000', 'threshold_green':'#008000'}
STAT_LAYOUT = {}
def setup_f258(profile='line'):
    local_fonts = Path.home()/'.local/share/fonts/microsoft-core-fonts-times'
    for path in local_fonts.glob('*'):
        if path.suffix.lower()=='.ttf':font_manager.fontManager.addfont(str(path))
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    label,tick,spine={'line':(8.685,7.237,.7237),
        'distribution':(8.481,5.654,1.0602),
        'histogram':(6.216,5.328,.666)}[profile]
    plt.rcParams.update({'font.family':'Times New Roman','font.size':tick,
        'axes.labelsize':label,'axes.labelweight':'bold','axes.labelpad':3,
        'xtick.labelsize':4.884 if profile=='histogram' else tick,
        'ytick.labelsize':tick,'legend.fontsize':tick,
        'axes.linewidth':spine,
        'lines.linewidth':1.4474,'pdf.fonttype':42,'ps.fonttype':42,
        'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
        'mathtext.it':'Times New Roman:italic',
        'mathtext.bf':'Times New Roman:bold'})
def finish_f258(ax, profile='line'):
    spine,grid={'line':(.7237,.579),'distribution':(1.0602,.5654),
        'histogram':(.666,.3552)}[profile]
    ax.set_axisbelow(True)
    ax.grid(True,ls='--' if profile=='line' else ':',lw=grid,
        color='.80' if profile=='line' else '.88')
    ax.tick_params(width=grid,length=2.8,pad=2)
    for side in ax.spines.values():
        side.set_color('.80' if profile=='line' else 'black')
        side.set_linewidth(spine)
def legend_f258(ax, **kwargs):
    legend=ax.legend(frameon=True,fancybox=False,framealpha=1,
        edgecolor='black',facecolor='white',borderpad=.35,
        handlelength=1.55,labelspacing=.25,handletextpad=.5,**kwargs)
    legend.get_frame().set_linewidth(.7237)
    return legend
def save_stat(fig,name,reference,out_dir):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    width,height=fig.get_size_inches()*72
    overflow=[]
    # Matplotlib keeps Text objects for ticks outside a log axis's view interval,
    # even though Axis.draw does not paint them. Exclude those undrawn labels.
    hidden_tick_labels=set()
    for ax in fig.axes:
        for axis in (ax.xaxis,ax.yaxis):
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc()<lo or tick.get_loc()>hi:
                    hidden_tick_labels.update((tick.label1,tick.label2))
    for artist in fig.findobj():
        if (isinstance(artist,matplotlib.text.Text) and artist.get_visible()
                and artist.get_text() and artist not in hidden_tick_labels):
            bbox=artist.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
            if bbox.x0<-.002 or bbox.y0<-.002 or bbox.x1>width/72+.002 or bbox.y1>height/72+.002:
                overflow.append(artist.get_text())
    assert not overflow,(name,overflow)
    STAT_LAYOUT[name]={'reference':reference,'canvas_bp':[width,height],
        'font':'Times New Roman','fixed_canvas':True,'all_text_inside_canvas':True,
        'axes_bp':[[x*width,y*height,w*width,h*height]
            for ax in fig.axes for x,y,w,h in [ax.get_position().bounds]],
        'coordinate_origin':'bottom left',
        'line_data':[[{'label':line.get_label(),'xy':line.get_xydata().tolist()}
            for line in ax.lines] for ax in fig.axes],
        'bar_heights':[[patch.get_height() for patch in ax.patches
            if isinstance(patch,matplotlib.patches.Rectangle)] for ax in fig.axes]}
    fig.savefig(Path(out_dir)/(name+'.pdf'),bbox_inches=None,pad_inches=0)
    fig.savefig(Path(out_dir)/(name+'.png'),dpi=240,bbox_inches=None,pad_inches=0)
    plt.close(fig)
