"""Build source/evidence cleanup inventory without executing model code."""
import ast
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'TBSI_gai/TBSI'
OUT = Path(__file__).resolve().parent
BASE = 'c0f1aff98792d604110a1e006a7d2cef4394734d'


def before(path):
    return subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=str(ROOT)).decode()


def block(path, symbol):
    text = before(path)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == symbol:
            return node.lineno, node.end_lineno, '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])
    raise ValueError('symbol not found: ' + path + ' ' + symbol)


with (PROJECT / 'experiments/experiment_asset_registry.csv').open(newline='') as stream:
    registry = list(csv.DictReader(stream))
all_refs = [(r['ID'], '\n'.join(r.values())) for r in registry]
code = []


def add(path, start, end, symbol, action, reason, dependency):
    text = before(path)
    snippet = '\n'.join(text.splitlines()[start-1:end])
    code.append(dict(path=path, baseline_start=start, baseline_end=end, symbol=symbol,
                     action=action, reason=reason, coupled_changes=dependency,
                     baseline_commit=BASE, snippet_sha256=hashlib.sha256(snippet.encode()).hexdigest()))


for path, starts in [
    ('TBSI_gai/TBSI/lib/models/tbsi_track/tbsi_track.py', [4, 5, 7]),
    ('TBSI_gai/TBSI/lib/models/tbsi_track/vit_tbsi_care.py', [21, 24, 25]),
]:
    lines = before(path).splitlines()
    for line in starts:
        add(path, line, line, lines[line-1], 'REMOVED_UNUSED_STDLIB_IMPORT',
            'AST name-use inventory found no load of this standard-library import', 'None; public tensor/model behavior unchanged')

layer = 'TBSI_gai/TBSI/lib/models/layers/tbsi_layer.py'
for symbol, dependency in [
    ('EarlyTargetMarginGate', 'constructor flags; margin branches; backbone config and build_tbsi_track arguments; historical initialization reset'),
    ('DiscriminativeEvidenceRouter', 'self.der construction; foreground masks; actor DER auxiliary loss; DER_ONLY optimizer/freeze path'),
    ('QueryConditionedCrossValue', 'query_conditioned construction/forward path; tracker-only flags and quick configs'),
    ('_condition_search_message', 'TBSILayer conditioned-message environment branch; historical T2S probes'),
    ('_early_gate_token_reliability', 'token_reliability_mode selection; cross gate multipliers; legacy quick3 configs'),
]:
    start,end,_ = block(layer,symbol)
    add(layer,start,end,symbol,'CANDIDATE_REMOVE_WITH_CALLERS_NOT_EXECUTED',
        'Outside the retained M1 / Early gate main path; historical behavior remains recoverable from checkpoint',dependency)
start,end,_=block(layer,'EarlyTemplateValueReconstruction')
add(layer,start,end,'EarlyTemplateValueReconstruction','HOLD_RESEARCH_ASSET_NOT_DELETE',
    'Latest audit retracts causal rejection; retaining the candidate does not mean selecting it for next training',
    'Future source/cross-only decision must be explicit; preserve class and checkpoint compatibility for now')
for start,end,symbol,dependency in [
    (658,733,'FCSV-R environment-only source redistribution','mode variables; later fcsv_cross_value_gate composition; old fixed-state runners'),
    (842,918,'template uncertainty experiment branches','participation/message_gate branches; uncertainty diagnostics; historical configs'),
    (1002,1057,'cross_scale counterfactual and collector path','cross-scale validation; analysis collectors; snapshot/replay tools'),
    (1347,1494,'query-conditioned / uncertainty / foreground-cross output dispatch','shared else contains Early gate native message; extract selected branches rather than deleting entire range'),
    (1495,1520,'TBSI_PRIORITY2_CAPTURE','capture readers; tensors used only for diagnostics'),
    (1523,1609,'receiver-centered Value residual','receiver mode env checks; relation diagnostics and fixed28 tools'),
]:
    add(layer,start,end,symbol,'CANDIDATE_EXTRACT_TO_LEGACY_NOT_EXECUTED',
        'Embedded experiment logic is mixed with the core forward; section requires dependency-aware extraction',dependency)

optional = {
 'adaptive_post_fusion_router.py': 'model constructor, forward_head, adaptive-only optimizer branch',
 'precenter_gated_residual.py': 'model constructor and forward_head target-context branch',
 'target_rank_gate.py': 'model build, actor target-rank loss, tracker diagnostics',
 'amcm_reliability.py': 'model fusion branches, actor AMCM losses, diagnostics, optimizer modes',
 'spatial_token_bridge.py': 'model fusion, actor mask generation, bridge-only training, configs',
 'search_conditioned_t2s_router.py': 'TBSILayer constructor/forward; STSER configs; actor masks',
 'early_utility_router.py': 'block2 capture/router; teacher maps; actor utility loss; optimizer and initializer',
 'early_search_residual.py': 'backbone constructor/forward; related scsu import',
 'post_tbsi_cross_modal_residual.py': 'backbone constructor/forward',
 'local_cross_modal_correspondence.py': 'backbone constructor/forward; explicit special initialization',
 'msha_screen_adapters.py': 'model screening adapters, screen-only training and config flags',
 'causal_tam_adapter.py': 'model/actor/tracker temporal state; preserve history until closure identified',
 'masked_template_hmoe.py': 'TBSILayer expert construction; historical template-expert configs',
 'pvl_adapter.py': 'backbone special RNG/init behavior; model/template/neck construction',
 'single_signal_prefusion.py': 'tracker-based monkey-patched prefusion routes; retained experiment evidence',
 'local_signal_post_fusion.py': 'tracker-based monkey-patched postfusion routes',
 'target_conditioned_local_search.py': 'No static import found inside lib; tools may use dynamic/source-file loading',
 'local_attender.py': 'Imported by target_conditioned_local_search.py; remove together only after tool audit',
}
for filename,dependency in optional.items():
    path='TBSI_gai/TBSI/lib/models/layers/'+filename
    text=before(path)
    add(path,1,len(text.splitlines()),filename,'CANDIDATE_LEGACY_MODULE_NOT_EXECUTED',
        'Separate from current M1/Early gate scope; absence of active scope is not proof of ineffective operator',dependency)

fields=list(code[0])
with (OUT/'code_cleanup_inventory.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(code)

active={
 'tbsi_e_mhsa_full_20260927.md':'Protected M1 report; preserve even without a literal registry path match',
 'm1_early_gate_bottleneck_audit_20261026.md':'Latest source audit; causal effect on performance remains unmeasured',
 'm1_early_gate_control_repeatability_20261026.md':'Raw same-control repeatability evidence',
 'm1_early_gate_value_recon_quick3_20261026.md':'Actual short-run result with corrected interpretation',
}
superseded={
 'm1_early_gate_optimization_analysis_20261026.md':'Contains old gate-focused plan and obsolete provenance-dependent attribution; historical evidence retained',
 'm1_msha_internal_input_interaction_audit_20261026.md':'Alternative QKV/input branch plan; not the selected next run',
 'm1_early_gate_value_interaction_recommendation_20261026.md':'Receiver-centered proposal; later fixed28 and subsequent audit supersede its action plan',
 'm1_early_gate_value_crosslayer_literature_scout_20261026.md':'ECLVR literature/proposal; does not establish trained benefit or current authorization',
 'm1_msha_2026_module_literature_scout_20261026.md':'Alternative literature branch; not a validated implementation or selected next experiment',
 'm1_early_gate_sequence_scene_attribution_20261026.md':'Header contains FSCIR/FCCR/FCSV priorities; gate-only proxy must not substitute full15 joint identity',
 'm1_dcvr_v1_v2_value_source_reallocation_20261026.md':'Historical FCSV-R direction; current proposal has not been chosen',
}
reports=[]
for path in sorted((PROJECT/'experiments/reports').glob('*.md')):
    rel=str(path.relative_to(PROJECT))
    ids=[identity for identity,text in all_refs if rel in text]
    if path.name in active: action='KEEP_CURRENT_ENTRY'; reason=active[path.name]
    elif path.name in superseded: action='ARCHIVE_ACTION_PLAN_KEEP_EVIDENCE'; reason=superseded[path.name]
    elif any(s in path.name for s in ['cleanup','weight_cleanup']):
        action='HISTORICAL_MAINTENANCE_RECORD';reason='Potential narrative consolidation; preserve deletion and provenance audit history'
    elif any(s in path.name for s in ['scout','literature','design','plan']):
        action='REFERENCE_ONLY_NOT_CURRENT_INSTRUCTION';reason='Design/literature label is a routing hint; no inference about validity or deletability'
    else: action='KEEP_HISTORICAL_EVIDENCE';reason='No verified duplicate; not safe to delete based on age or negative metrics'
    reports.append(dict(path=str(path.relative_to(ROOT)),action=action,reason=reason,
                        direct_registry_references=len(ids),asset_ids=';'.join(ids),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        physical_delete='NO'))
with (OUT/'report_cleanup_inventory.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(reports[0]));writer.writeheader();writer.writerows(reports)

configs=[]
protected={'TBSI_E_MHSA_FULL_4090D_seed42.yaml'}
for p in sorted((PROJECT/'experiments/tbsi_track').glob('*.yaml')):
    if p.name in protected or p.name.startswith(('v1.0.1-','v1.0.2-','v1.0.3-')):action='KEEP_PROTECTED_ANCHOR'
    elif p.name.startswith('M1_EARLY_CONDITIONED_MSHA_GATE_FULL15_'):action='KEEP_PARENT_IDENTITY'
    else:action='REVIEW_HISTORICAL_CONFIG_KEEP_PENDING_DEPENDENCY_AUDIT'
    configs.append({'path':str(p.relative_to(ROOT)),'action':action,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'physical_delete':'NO'})
with (OUT/'config_cleanup_inventory.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(configs[0]));writer.writeheader();writer.writerows(configs)

summary={'baseline_commit':BASE,'code_items':len(code),'code_actions':dict(Counter(r['action'] for r in code)),
         'reports':len(reports),'report_actions':dict(Counter(r['action'] for r in reports)),
         'report_exact_duplicate_groups':0,'configs':len(configs),
         'model_tests_run':False,'training_started':False,
         'code_line_numbers':'All ranges refer to baseline commit, not shifted working tree lines'}
(OUT/'inventory_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
