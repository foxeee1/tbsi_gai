#!/usr/bin/env python3
"""Rebuild the machine registry from its authoritative Markdown source."""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / 'experiments/实验资产数据库.md'
CSV = ROOT / 'experiments/experiment_asset_registry.csv'
DAMAGED = ROOT / 'experiments/snapshots/experiment_asset_registry_truncated_20261014.csv'
FIELDS = ['ID','Date','Parent','CommitOrSnapshot','Config','Seed','Dataset','EvalScope',
          'AUC','PR','NPR','OP50','OP75','FPS','GPU','HardwareConfig','GPUHours',
          'Checkpoint','AssumptionOrHypothesis','Conclusion','Status','Evidence','Notes']

def clean(value):
    return re.sub(r'\s+', ' ', value.strip().strip('`'))

def first_number(text):
    m = re.search(r'(?<![A-Za-z])(-?\d+(?:\.\d+)?)', text.replace(',', ''))
    return m.group(1) if m else 'NA'

def row_from_markdown(line_no, raw, cells):
    ident = clean(cells[0])
    date_match = re.search(r'(20\d{6})$', ident)
    date = date_match.group(1) if date_match else 'NA'
    status_index = 7 if len(cells) == 9 else 5 if len(cells) == 7 else 3 if len(cells) == 5 else -1
    status_text = clean(cells[status_index]) if status_index >= 0 else 'NA'
    conclusion_index = 8 if len(cells) == 9 else 6 if len(cells) == 7 else 3 if len(cells) == 5 else -1
    conclusion = clean(cells[conclusion_index]) if conclusion_index >= 0 else 'NA'
    evidence_index = 8 if len(cells) == 9 else 6 if len(cells) == 7 else 4 if len(cells) == 5 else -1
    evidence = clean(cells[evidence_index]) if evidence_index >= 0 else 'NA'
    metrics_cell = cells[5] if len(cells) == 9 else cells[3] if len(cells) == 7 else ''
    op_cell = cells[6] if len(cells) == 9 else cells[4] if len(cells) == 7 else ''
    auc = pr = npr = op50 = op75 = fps = 'NA'
    if metrics_cell:
        nums = re.findall(r'(?<![A-Za-z])(-?\d+(?:\.\d+)?)', metrics_cell)
        if len(nums) >= 3:
            auc, pr, npr = nums[:3]
    if op_cell:
        nums = re.findall(r'(?<![A-Za-z])(-?\d+(?:\.\d+)?)', op_cell)
        if len(nums) >= 2:
            op50, op75 = nums[:2]
        fps_match = re.search(r'FPS[^\d-]*(-?\d+(?:\.\d+)?)', op_cell)
        if fps_match:
            fps = fps_match.group(1)
    all_text = ' '.join(cells)
    config_match = re.search(r'([\w./-]+\.yaml)', all_text)
    seed_match = re.search(r'\bseed\s*[:=]?\s*(42|3407)\b', all_text, re.I)
    seed = clean(cells[3]) if len(cells) == 9 and cells[3].strip().isdigit() else seed_match.group(1) if seed_match else 'NA'
    gpu_match = re.search(r'(NVIDIA GeForce RTX [^;；,，]+)', all_text)
    gpu = gpu_match.group(1).strip() if gpu_match else 'NVIDIA GeForce RTX 4090 D' if '4090D' in all_text or '4090 D' in all_text else 'NA'
    hours_match = re.search(r'(?:GPUHours|GPU\s*小时|GPUHours约)\s*[:：]?\s*([0-9.]+)', all_text, re.I)
    checkpoint_match = re.search(r'(checkpoint[^;；|]*|权重[^;；|]*)(?:[;；|]|$)', all_text, re.I)
    dataset_cells = [clean(c) for c in cells[1:-1] if re.search(r'LasHeR|dataset|testingset|train-side', c, re.I)]
    scope = '；'.join(dataset_cells) if dataset_cells else 'NA'
    parent = clean(cells[1]) if len(cells) > 1 else 'NA'
    provenance_index = 2 if len(cells) >= 9 else 1
    provenance = clean(cells[provenance_index]) if len(cells) > provenance_index else 'NA'
    return {
        'ID':ident,'Date':date,'Parent':parent,'CommitOrSnapshot':provenance,
        'Config':config_match.group(1) if config_match else 'NA','Seed':seed,
        'Dataset':'LasHeR' if 'LasHeR' in all_text else 'NA','EvalScope':scope,
        'AUC':auc,'PR':pr,'NPR':npr,'OP50':op50,'OP75':op75,'FPS':fps,
        'GPU':gpu,'HardwareConfig':'NA','GPUHours':hours_match.group(1) if hours_match else 'NA',
        'Checkpoint':checkpoint_match.group(1) if checkpoint_match else 'NA',
        'AssumptionOrHypothesis':'NA','Conclusion':conclusion,'Status':status_text,
        'Evidence':evidence,'Notes':'Reconstructed from authoritative Markdown line {}; raw={}'.format(line_no,clean(raw))
    }

def main():
    records = {}
    for line_no, raw in enumerate(MD.read_text().splitlines(), 1):
        if not raw.startswith('|'):
            continue
        cells = [x.strip() for x in raw.strip().strip('|').split('|')]
        if not cells:
            continue
        ident = clean(cells[0])
        if re.match(r'^[A-Z][A-Z0-9_-]{7,}$', ident):
            records[ident] = row_from_markdown(line_no, raw, cells)
    # Keep valid rows from the truncated file when the Markdown does not contain them.
    if DAMAGED.exists():
        with DAMAGED.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row.get('ID') and row['ID'] not in records and None not in row:
                    records[row['ID']] = {field: row.get(field, 'NA') or 'NA' for field in FIELDS}
    # Restore the bounded predictor records that were appended after the Markdown snapshot.
    predictor = {
        'ID':'M1-BOUNDED-READOUT-PREDICTOR-20261014','Date':'2026-10-14',
        'Parent':'TBSI_E_MHSA_FULL_4090D_seed42 epoch15','CommitOrSnapshot':'working-tree-20261014; tools/fit_m1_bounded_readout_predictor_20261014.py',
        'Config':'experiments/tbsi_track/TBSI_E_MHSA_FULL_4090D_seed42.yaml','Seed':'42',
        'Dataset':'LasHeR train-side','EvalScope':'72 sequences / 144 fixed states; sequence-held-out decoded replay',
        'AUC':'NA','PR':'NA','NPR':'NA','OP50':'NA','OP75':'NA','FPS':'NA',
        'GPU':'NVIDIA GeForce RTX 4090 D','HardwareConfig':'1x RTX4090D; Python3.8.20/PyTorch2.0.0+cu118; fixed-state replay',
        'GPUHours':'NA','Checkpoint':'protected M1 epoch15 checkpoint',
        'AssumptionOrHypothesis':'Fit bounded offset/size prediction with no-op fallback using deployment-visible peak margin',
        'Conclusion':'threshold 0.08 coverage 51.4%; selected utility +0.05963; harmful 0%; threshold 0.04 coverage 81.9%; harmful 1.69%',
        'Status':'DIAGNOSTIC-COMPLETE / BOUNDED-PREDICTOR-PROMISING / CLOSED-LOOP-PROXY-REQUIRED',
        'Evidence':'experiments/reports/m1_bounded_readout_predictor_20261014.md; output/experiments/M1_BOUNDED_READOUT_PREDICTOR_20261014_retry3/summary.json',
        'Notes':'Recovered from preserved experiment notes after registry repair.'}
    records[predictor['ID']] = predictor
    # Keep unique IDs and deterministic order by date and original insertion order.
    tmp = CSV.with_suffix('.csv.rebuilt')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records.values())
    with tmp.open(newline='') as stream:
        check = list(csv.reader(stream))
    bad = [i+1 for i,row in enumerate(check) if len(row) != len(FIELDS)]
    ids = [row[0] for row in check[1:]]
    if bad or len(ids) != len(set(ids)):
        raise RuntimeError('rebuild validation failed: bad rows={}, ids={}, unique={}'.format(bad[:5],len(ids),len(set(ids))))
    tmp.replace(CSV)
    print('rebuilt {} unique records with {} columns'.format(len(ids),len(FIELDS)))

if __name__ == '__main__':
    main()
