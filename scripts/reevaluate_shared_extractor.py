"""Rescore all saved pilot comparisons with one extractor; preserve prior runs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from ocr_common import ROOT, write_json
from evaluate_six_critical_fields import (
    EXTRACTOR_VERSION, FAMILIES, SOURCES, SOURCE_SHA256, SAVED, annotations, extract, sha)
from evaluate_paddleocr_v4 import read, write_csv, PAGE_COUNTS
from evaluate_unlimited_ocr import verified_baselines, verify_unlimited, METHOD
from evaluate_claude_vlm import verify_claude, MODEL as CLAUDE

sys.path.insert(0, str(ROOT / 'standalone_ocr_eval'))
from ocr_eval import evaluate_benchmark
from ocr_eval.field_values import field_text
from ocr_eval.profiles import fingerprint

SCENARIOS = [
    ('six_critical_field_evaluation', 'results.json'),
    ('paddleocr_v4_experiment', 'critical_fields.json'),
    ('openai_vlm_experiment', 'critical_fields.json'),
    ('claude_vlm_experiment', 'critical_fields.json'),
    ('unlimited_ocr_experiment', 'critical_fields.json'),
    ('unlimited_ocr_rotation_experiment', 'critical_fields.json'),
]


def verified_inputs():
    """Validate native OCR and frozen inputs independently of old field rules."""
    manifest, base, _, _, _, _ = verified_baselines(require_current_extractor=False)
    paddle_folder = Path(manifest['baseline_manifest']).parent
    paddle = read(paddle_folder / 'input.json')
    six = read(SAVED / 'input.json')
    for doc, methods in six.items():
        for method, raw in methods.items():
            if raw != paddle[doc][method]:
                raise ValueError('Six-method text differs from verified baseline')
    expected = {'six_critical_field_evaluation': six,
                'paddleocr_v4_experiment': paddle, 'openai_vlm_experiment': base}
    records = {}
    for scenario, ocr_folder in [
            ('unlimited_ocr_experiment', 'unlimited-ocr-bf16'),
            ('unlimited_ocr_rotation_experiment', 'unlimited-ocr-bf16-rotated')]:
        root = ROOT / 'output' / ocr_folder
        path = root / 'manifest.json'
        native = read(path)
        prior = read(ROOT / 'output' / scenario / 'run_manifest.json')
        if sha(path) != prior['ocr_manifest_sha256'] or native['failures'] or len(native['documents']) != 3:
            raise ValueError('Unlimited-OCR manifest changed or incomplete')
        data = deepcopy(base)
        for doc, name in SOURCES.items():
            for excluded in ['openai-6', 'fable-5-1']:
                data[doc].pop(excluded, None)
            record, = [r for r in native['documents'] if r['source'] == name]
            responses, _ = verify_unlimited(record, root, ROOT / 'data' / name, PAGE_COUNTS[doc])
            data[doc][METHOD] = {'pages': ['\n\n'.join(r['markdown'] for r in responses)], 'input_format': 'markdown'}
        expected[scenario] = data
        records[str(path)] = sha(path)
    claude_folder = ROOT / 'output/claude_vlm_experiment'
    prior = read(claude_folder / 'run_manifest.json')
    path = Path(prior['ocr_manifest'])
    if sha(path) != prior['ocr_manifest_sha256']:
        raise ValueError('Claude manifest changed')
    native = read(path)
    selected = prior['included_documents']
    if set(selected) != {'board', 'trade'}:
        raise ValueError('Unexpected Claude comparison scope')
    data = {doc: deepcopy(base[doc]) for doc in selected}
    for doc in selected:
        for excluded in ['openai-6', 'fable-5-1']:
            data[doc].pop(excluded, None)
        record, = [r for r in native['documents'] if r['source'] == SOURCES[doc]]
        responses, _ = verify_claude(record, path.parent, doc)
        data[doc][CLAUDE] = {'pages': ['\n\n'.join(r['markdown'] for r in responses)], 'input_format': 'markdown'}
    expected['claude_vlm_experiment'] = data
    records[str(path)] = sha(path)
    return expected, records


def rescore(name, score_file, data, output, source_profiles, source_refs, consistent):
    original = ROOT / 'output' / name
    old = read(original / score_file)
    profiles, refs, predictions = [read(original / p) for p in ['profiles.json', 'references.json', 'predictions.json']]
    manifest = read(original / 'run_manifest.json')
    for key, value in [('profile_sha256', profiles), ('reference_sha256', refs), ('predictions_sha256', predictions)]:
        if fingerprint(value) != old['settings'][key]:
            raise ValueError(f'{name}: old pilot artifact changed: {key}')
    if profiles != source_profiles or refs != {**source_refs, 'documents': {
            doc: source_refs['documents'][doc] for doc in data}}:
        raise ValueError(f'{name}: references or profiles differ from frozen source schema')
    if name == 'six_critical_field_evaluation':
        if sha(Path(manifest['input_path'])) != manifest['input_sha256']:
            raise ValueError('Six-method input hash changed')
    else:
        if read(original / 'input.json') != data:
            raise ValueError(f'{name}: text differs from verified native OCR')
        if manifest.get('input_sha256') and sha(original / 'input.json') != manifest['input_sha256']:
            raise ValueError(f'{name}: comparison input hash changed')
    new_predictions = deepcopy(predictions)
    recomputed_old = evaluate_benchmark(profiles, refs, predictions, split='validation')
    for method, scored in recomputed_old['methods'].items():
        if (scored['metrics'] != old['methods'][method]['metrics'] or
                scored['score'] != old['methods'][method]['score']):
            raise ValueError('Previous reported score differs from frozen predictions')
    new_predictions['methods'] = {}
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    for method, previous in predictions['methods'].items():
        docs = {}
        for doc in data:
            record = data[doc][method]
            # Preserve the historical Markdown convention for the six old labels.
            fmt = 'markdown' if method in FAMILIES else record.get('input_format', 'markdown')
            text = field_text('\n\n'.join(record['pages']), fmt)
            captures = extract(doc, text)
            for item in captures:
                a, b = item['decoded_span']
                if item['value'] != text[a:b]:
                    raise ValueError('Captured OCR value was modified')
            key = (method, doc, sha_text(text))
            if key in consistent and consistent[key] != captures:
                raise ValueError('Same OCR text received different extraction rules')
            consistent[key] = captures
            docs[doc] = {'fields': captures}
            decoded = folder / 'decoded' / method
            decoded.mkdir(parents=True, exist_ok=True)
            (decoded / (doc + '.txt')).write_text(text, encoding='utf-8')
        new_predictions['methods'][method] = {'family': previous['family'], 'documents': docs,
            'provenance': {'extractor_version': EXTRACTOR_VERSION, 'previous_provenance': previous.get('provenance'),
                'extraction': 'Shared inline and explicit equal-length invoice column rules; first invoice; raw spans; no value repair.'}}
    result = evaluate_benchmark(profiles, refs, new_predictions, split='validation')
    result.update(extractor_version=EXTRACTOR_VERSION, meaning=old['meaning'],
                  pilot_limitations=old['pilot_limitations'])
    rows, audit, changes = [], [], []
    for rank in result['ranking']:
        method = rank['method']; new = result['methods'][method]; before = old['methods'][method]
        rows.append({**rank, 'correct': new['metrics']['critical_correct'], 'total': new['metrics']['critical_fields'],
                     'previous_correct': before['metrics']['critical_correct'], 'previous_score': before['score'],
                     **{d + '_correct': r['counts']['critical_correct'] for d, r in new['documents'].items()}})
        for doc, scored in new['documents'].items():
            previous_fields = {f['field']: f for f in before['documents'][doc]['field_results']}
            for f in scored['field_results']:
                prior = previous_fields[f['field']]
                if f['reference_value'] != prior['reference_value']:
                    raise ValueError('Reference value changed')
                row = {'method': method, 'document': doc, 'field': f['field'],
                    'source_page': f['reference_location']['page'], 'reference': f['reference_value'],
                    'previous_prediction': prior['predicted_value'], 'prediction': f['predicted_value'],
                    'previous_result': prior['reason'], 'result': f['reason']}
                audit.append(row)
                if prior['reason'] != f['reason'] or prior['predicted_value'] != f['predicted_value']:
                    changes.append(row)
    for file, value in [('input', data), ('profiles', profiles), ('references', refs),
                        ('predictions', new_predictions), ('critical_fields', result), ('field_changes', changes)]:
        write_json(folder / (file + '.json'), value)
    write_csv(folder / 'critical_field_ranking.csv', rows)
    write_csv(folder / 'field_audit.csv', audit)
    if changes:
        write_csv(folder / 'field_changes.csv', changes)
    write_json(folder / 'run_manifest.json', {'extractor_version': EXTRACTOR_VERSION,
        'previous_extractor_version': old['extractor_version'], 'included_documents': list(data),
        'previous_manifest_sha256': sha(original / 'run_manifest.json'),
        'previous_result_sha256': sha(original / score_file), 'input_sha256': sha(folder / 'input.json'),
        'profile_sha256': fingerprint(profiles), 'reference_sha256': fingerprint(refs),
        'extractor_script_sha256': sha(ROOT / 'scripts/evaluate_six_critical_fields.py'),
        'evaluation_script_sha256': sha(Path(__file__)), 'source_sha256': SOURCE_SHA256})
    return {'scenario': name, 'documents': list(data), 'rows': rows, 'changes': changes}


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / f'output/extractor_v{EXTRACTOR_VERSION}_evaluation')
    args = parser.parse_args()
    output = args.output.resolve()
    if any(output == ROOT / 'output' / name for name, _ in SCENARIOS):
        parser.error('Use a separate output directory to preserve old comparisons')
    data, native = verified_inputs()
    profiles, refs = annotations()
    consistent, scenarios = {}, []
    for name, score in SCENARIOS:
        print(f'Reextracting every method in {name} with version {EXTRACTOR_VERSION}', flush=True)
        scenarios.append(rescore(name, score, data[name], output, profiles, refs, consistent))
    write_json(output / 'summary.json', scenarios)
    write_json(output / 'run_manifest.json', {'extractor_version': EXTRACTOR_VERSION,
        'extractor_script_sha256': sha(ROOT / 'scripts/evaluate_six_critical_fields.py'),
        'evaluation_script_sha256': sha(Path(__file__)), 'verified_native_manifests': native,
        'source_sha256': SOURCE_SHA256, 'scenarios': [s['scenario'] for s in scenarios]})
    lines = [f'# Shared extractor version {EXTRACTOR_VERSION}: all saved pilot comparisons', '',
        'Reextracted every method from verified saved OCR with the same updated rules. No new OCR requests. '
        'Previous outputs, reference values, scoring policies and document scopes are preserved. '
        'The new rule binds contiguous unique DOC NO/DATE[/GST NO] labels to an equal number of '
        'colon-prefixed value rows by order. Missing/extra rows remain unextracted; wrong values remain wrong. '
        'Direct label/value captures take precedence. Captures retain exact raw decoded spans.', '',
        'This is a post-hoc exploratory OCR-plus-extraction pilot on selected image-checked references; '
        'it is not held-out or independently adjudicated accuracy. Whole-text agreement is unchanged '
        'because the OCR text and peer sets are unchanged.', '']
    for scenario in reversed(scenarios):
        lines += [f'## {scenario["scenario"]}', '', f'Documents: {", ".join(scenario["documents"])}. '
            'Claude includes only board/trade (21 fields), so its scores cannot be compared directly with '
            'the full three-document (27-field) scores.' if scenario['scenario'] == 'claude_vlm_experiment'
            else f'Documents: {", ".join(scenario["documents"])}; 27 fields.', '',
            '| Method | Before /100 | After /100 | Correct before → after | Board | CSA | Trade |',
            '|---|---:|---:|---|---:|---:|---:|']
        for row in scenario['rows']:
            lines.append(f'| {row["method"]} | {row["previous_score"]:.2f} | {row["critical_field_score"]:.2f} | '
                f'{row["previous_correct"]} → {row["correct"]}/{row["total"]} | '
                + ' | '.join(str(row.get(d + '_correct', '—')) for d in ['board', 'csa', 'trade']) + ' |')
        lines += ['', f'Changed field outcomes: {len(scenario["changes"])}. '
                  f'See `{scenario["scenario"]}/field_audit.csv` for before/after predictions.', '']
    lines += ['## Reproduce', '', '```bash',
        'conda run --no-capture-output -n myenv3.13 python scripts/reevaluate_shared_extractor.py', '```', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(scenarios, indent=2))
    print(output / 'report.md')


if __name__ == '__main__':
    main()
