"""Check manuscript structure and transcribed values; does not run experiments."""
from pathlib import Path
import csv
import difflib
import hashlib
import json
import re
import shutil

HERE = Path(__file__).resolve().parent
PROJECT = Path('/root/autodl-tmp/TBSI_gai/TBSI')
tex = (HERE / 'latex/manuscript_revised.tex').read_text()
bib = (HERE / 'latex/references.bib').read_text()
errors = []
checks = []


def check(name, condition):
    checks.append({'check': name, 'passed': bool(condition)})
    if not condition:
        errors.append(name)


def strip_comments(text):
    return '\n'.join(re.split(r'(?<!\\)%', line, maxsplit=1)[0]
                     for line in text.splitlines())


clean = strip_comments(tex)
stack = []
for match in re.finditer(r'\\(begin|end)\{([^}]+)\}', clean):
    action, environment = match.groups()
    if action == 'begin':
        stack.append(environment)
    elif not stack or stack.pop() != environment:
        errors.append('Mismatched environment: ' + environment)
check('LaTeX environments balanced', not stack and not any('environment:' in x for x in errors))
depth = 0
for match in re.finditer(r'(?<!\\)[{}]', clean):
    depth += 1 if match.group() == '{' else -1
    if depth < 0:
        errors.append('Closing brace before opening brace')
        break
check('Unescaped braces balanced', depth == 0)
check('Inline math delimiters paired', len(re.findall(r'(?<!\\)\$', clean)) % 2 == 0)
labels = re.findall(r'\\label\{([^}]+)\}', clean)
refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', clean)
check('Unique labels', len(labels) == len(set(labels)))
check('All cross-references resolve', set(refs) <= set(labels))
keys = set(re.findall(r'@\w+\s*\{([^,]+),', bib))
cites = set()
for group in re.findall(r'\\cite\{([^}]+)\}', clean):
    cites.update(group.split(','))
check('All citation keys resolve', cites <= keys)
check('No proposed DCVR-v2 method carried into revised prose', 'DCVR-v2' not in clean)
check('No result placeholder figures', not any(x in clean for x in ['图件预留', '待实验', '待验证']))

table_columns = [8, 6, 7, 6, 4, 8, 8]
tables = re.findall(r'\\begin\{tabular\}(.*?)\\end\{tabular\}', clean, re.S)
check('Expected seven tables', len(tables) == len(table_columns))
for index, (table, columns) in enumerate(zip(tables, table_columns), 1):
    rows = [x for x in table.split('\\\\') if '&' in x]
    check('Table %d column counts' % index,
          all(len(re.findall(r'(?<!\\)&', row)) == columns - 1 for row in rows))

expected = {
    'EG-EVDR-ALIGNED-FULL15-SOT-S42-20261027': [56.0956, 69.6379, 66.0366, 67.8290, 48.1135],
    'EG-SPATIAL-R-SEP-FULL15-TRAIN-S42-20261030': [56.4061, 70.1943, 66.2966, 68.2390, 48.2813],
    'EG-EARLY-GATE-UTILITY-FULL15-UTILITY-S42-20261031': [56.5590, 70.3386, 66.5281, 68.4519, 48.2914],
    'EG-SEARCH-CONTEXT-FULL15-C-FULL244-S42-20261031': [56.6081, 70.2697, 66.5861, 68.6029, 48.4569],
    'EG-C-EARLY-LR-DROP-U936-S42-20261101': [56.8305, 70.4502, 66.8347, 68.8178, 49.0406],
    'EG-C-LAYER-ALIGNED-CONTEXT-U936-S42-20261101': [56.4307, 69.9619, 66.4300, 68.3135, 48.5553],
}
with (PROJECT / 'experiments/experiment_asset_registry.csv').open() as stream:
    registry = {r['ID']: r for r in csv.DictReader(stream)}
fields = ['AUC', 'PR', 'NPR', 'OP50', 'OP75']
for asset_id, values in expected.items():
    check('Registry transcription: ' + asset_id,
          asset_id in registry and all(abs(float(registry[asset_id][k]) - v) < 0.000051
                                      for k, v in zip(fields, values)))
    check('Manuscript values present: ' + asset_id,
          all(f'{v:.4f}' in clean for v in values))

ours = expected['EG-C-EARLY-LR-DROP-U936-S42-20261101']
for name, baseline, target in [
    ('B0', [54.82, 68.53, 64.59, 66.52, 45.80], [2.01, 1.92, 2.24, 2.30, 3.24]),
    ('M1', [56.207716635295306, 70.20728866200643, 66.1979574361281,
            68.33389396359762, 47.13491503544764], [0.62, 0.24, 0.64, 0.48, 1.91]),
]:
    check('Rounded delta arithmetic: ' + name,
          [round(x-y, 2) for x, y in zip(ours, baseline)] == target)
check('Training sample/update arithmetic', 59904 * 15 == 898560 and 7020 * 16 * 8 == 898560)

original = (HERE / 'original/manuscript_original.tex').read_text().splitlines(keepends=True)
revised = tex.splitlines(keepends=True)
(HERE / 'manuscript_changes.diff').write_text(''.join(difflib.unified_diff(
    original, revised, fromfile='original/manuscript_original.tex',
    tofile='latex/manuscript_revised.tex')))
result = {
    'checks': checks,
    'errors': errors,
    'latex_engine_available': {k: shutil.which(k) for k in ['xelatex', 'latexmk', 'bibtex']},
    'compiled': False,
    'render_inspected': False,
    'manuscript_sha256': hashlib.sha256(tex.encode()).hexdigest(),
    'bibliography_sha256': hashlib.sha256(bib.encode()).hexdigest(),
}
(HERE / 'validation_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
report = ['# 检查结果', '',
          f'静态/数值核对：{sum(c["passed"] for c in checks)}/{len(checks)} 项通过。', '',
          '已核对 LaTeX 环境与花括号、数学分隔符、交叉引用、引用键、七张表的列数、六组新候选五指标与资产 CSV 的转录一致性，以及 B0/M1 差值和训练预算算术。', '',
          '结构检查不等于 LaTeX 编译。当前未安装 XeLaTeX、LaTeX/BibTeX 工具链，未生成 PDF，也未检查最终浮动体位置、溢出、字体和分页。', '',
          '参考文献 DOI 元数据在线取得；TBSI 与 LasHeR 作者 README 已读取。TPAMI 延伸工作未阅读全文，只作题名所支持的一般背景归属。', '',
          '没有执行新训练、推理、评测、bootstrap 或其他研究诊断。论文中实验读数均转引已有产物。', '',
          '本次只在 paper_revision_20261102 下新增写作交付文件；实验代码、历史配置、checkpoint、结果及双资产表未修改。', '',
          '详细机器输出：`validation_results.json`。']
if errors:
    report += ['', '失败项：'] + ['- ' + e for e in errors]
(HERE / 'VALIDATION.md').write_text('\n'.join(report) + '\n')
print(json.dumps({'passed': sum(c['passed'] for c in checks), 'total': len(checks), 'errors': errors}, ensure_ascii=False))
raise SystemExit(bool(errors))
