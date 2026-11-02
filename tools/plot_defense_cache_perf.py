"""Regenerate latency panels from saved timings; no models or network calls."""
import argparse
import json
from pathlib import Path
from paper_figures.result_panels import plot_latency


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();data=json.loads(args.result.read_text())
    assert all(len(row['samples_seconds_per_1000_tokens'])==100 for row in data['perf'].values())
    args.out.mkdir(parents=True,exist_ok=True);plot_latency(data['perf'],args.out)
    print('Latency panels regenerated from all retained samples.')
