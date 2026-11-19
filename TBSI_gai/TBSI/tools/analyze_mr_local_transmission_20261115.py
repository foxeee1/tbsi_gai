#!/usr/bin/env python3
"""Summarize frozen interventions with sample/sequence units kept explicit."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/experiments/MSHA-MR-LOCAL-TRANSMISSION-AUDIT-S42-20261115'


def records(frame):
    return json.loads(frame.reset_index().to_json(orient='records'))


def summarize(directory):
    b=pd.read_csv(directory/'baseline.csv')
    r=pd.read_csv(directory/'representation.csv')
    i=pd.read_csv(directory/'interventions.csv')
    g=pd.read_csv(directory/'gradients.csv')
    f=pd.read_csv(directory/'finite_difference.csv')
    result=dict(completed=json.loads((directory/'completed.json').read_text()),
        baseline=records(b.groupby('split').agg(frames=('sample','size'),target_center_inside=('center_inside','sum'),mean_iou=('iou','mean'))),
        representation=records(r.groupby(['split','layer','region'])[['detail_energy_fraction','source_v_detail_energy_fraction','tokens']].mean()),
        effects=records(i.groupby(['split','action'])[['iou_delta','margin_delta','loss_delta','hann_peak_changed','raw_peak_changed',
            'box_max_pixel_delta','edit_relative_full_message','projection_relative_delta','fusion_relative_delta','next_search_relative_delta',
            'final_search_relative_delta','score_relative_delta','hann_relative_delta']].mean()),
        region_effects=records(i.groupby(['split','layer','region','action'])[['iou_delta','margin_delta','loss_delta','hann_peak_changed','final_search_relative_delta']].mean()),
        gradient=records(g.groupby(['split','action']).directional_derivative.agg(['mean','median','min','max'])),
        finite_difference=records(f.groupby('epsilon')[['absolute_error','symmetric_relative_error','sign_matches']].mean()),
        finite_difference_resolved=records(f[f.analytic.abs()>=1e-5].groupby('epsilon')[['absolute_error','symmetric_relative_error','sign_matches']].mean()),
        negative_loss_fraction=records(i.groupby(['split','action']).loss_delta.apply(lambda x:float((x.dropna()<-1e-6).mean())).to_frame('fraction')),
        positive_loss_fraction=records(i.groupby(['split','action']).loss_delta.apply(lambda x:float((x.dropna()>1e-6).mean())).to_frame('fraction')),
        interpretation='Within-checkpoint fixed-crop exploratory effects. No closed-loop AUC, independent holdout, training improvement or cross-seed claim.')
    peak=i.groupby(['split','action']).apply(lambda x:pd.Series(dict(interventions=len(x),hann_peak_changes=int(x.hann_peak_changed.sum()),
        frames_with_any_change=int(x.groupby('sample').hann_peak_changed.any().sum()),frames=x['sample'].nunique())))
    result['peak_changes']=records(peak)
    i.groupby(['split','sequence','action'])[['iou_delta','margin_delta','loss_delta','hann_peak_changed','box_max_pixel_delta']].mean().to_csv(directory/'per_sequence_effects.csv')
    (directory/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result


def main():
    results={'quick_high':summarize(RUN)}
    for name in ('full_mr_g1','quick_highest_precision'):
        if (RUN/name/'completed.json').exists():
            results[name]=summarize(RUN/name)
    (RUN/'all_summaries.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    for name,value in results.items():
        print(name)
        print(pd.DataFrame(value['baseline']).to_string(index=False))
        print(pd.DataFrame(value['representation']).to_string(index=False))
        print(pd.DataFrame(value['effects'])[['split','action','iou_delta','margin_delta','loss_delta','final_search_relative_delta','hann_peak_changed']].to_string(index=False))
        print('Finite difference:',value['finite_difference'])
        print('Resolved finite difference:',value['finite_difference_resolved'])


if __name__=='__main__':main()
