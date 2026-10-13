"""Compare a complete PP-OCRv4 run with the six frozen OCR outputs and pilot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from ocr_common import ROOT, write_json
from evaluate_six_critical_fields import (FAMILIES, SOURCES, SOURCE_SHA256,
                                         SAVED, OUT as PILOT, annotations, extract, sha,
                                         EXTRACTOR_VERSION)

sys.path.insert(0, str(ROOT / 'standalone_ocr_eval'))
from ocr_eval import evaluate, evaluate_benchmark, consensus_family_sensitivity
from ocr_eval.profiles import fingerprint
from ocr_eval.field_values import field_text

METHOD = 'paddleocr-v4'
PAGE_COUNTS = {'csa': 14, 'board': 4, 'trade': 30}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_run(record, source, root, expected_pages):
    """Reject partial, reordered, stale or modified OCR before computing scores."""
    if record['source_sha256'] != sha(source):
        raise ValueError(f'Source hash mismatch: {source}')
    folder = (root / record['run']).resolve()
    if not folder.is_relative_to(root.resolve()):
        raise ValueError('Run path escapes output root')
    metadata = read(folder / 'source.json')
    if metadata['source'] != record['source'] or metadata['backend'] != METHOD:
        raise ValueError('Source/backend metadata mismatch')
    config = {k: v for k, v in metadata.items() if k not in ('source', 'responses')}
    digest = hashlib.sha256(source.read_bytes())
    digest.update(json.dumps(config, sort_keys=True).encode())
    if folder.name != digest.hexdigest()[:16]:
        raise ValueError('Source/settings directory hash mismatch')
    paths = sorted(folder.glob('response-*.json'))
    if len(paths) != expected_pages or metadata['responses'] != expected_pages or record['pages'] != expected_pages:
        raise ValueError('Incomplete or extra page checkpoints')
    for path in [folder / 'source.json', *paths]:
        if record['artifact_sha256'].get(path.relative_to(root.resolve()).as_posix()) != sha(path):
            raise ValueError(f'Artifact hash mismatch: {path}')
    responses = []
    for page, path in enumerate(paths, 1):
        r = read(path)
        if r['page'] != page or path.name != f'response-{page:04d}.json':
            raise ValueError('Page order mismatch')
        if r['input_format'] != 'plain' or r['text'] != '\n'.join(r['raw']['rec_texts']):
            raise ValueError('Text differs from native PaddleOCR output')
        if not (len(r['raw']['rec_texts']) == len(r['raw']['rec_scores']) == len(r['raw']['rec_polys'])):
            raise ValueError('Text/score/polygon count mismatch')
        responses.append(r)
    return responses, metadata


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ocr-output', type=Path, default=ROOT / 'output/paddleocr-v4')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/paddleocr_v4_experiment')
    args = parser.parse_args()
    output, ocr_root = args.output.resolve(), args.ocr_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = read(SAVED / 'input.json')
    prior = read(SAVED / 'with_ontology/results.json')
    provenance = prior['provenance']
    if sha(SAVED / 'input.json') != provenance['filtered_input_sha256']:
        raise ValueError('Frozen six-method input changed')
    verified = {}
    for record in provenance['artifacts']:
        for path, expected in record.get('artifact_sha256', {}).items():
            if sha(Path(path)) != expected:
                raise ValueError(f'Baseline OCR artifact changed: {path}')
            verified[path] = expected
    profiles, refs = annotations()  # Includes fixed source PDF hash checks.
    baseline = read(PILOT / 'results.json')
    if baseline.get('extractor_version') != EXTRACTOR_VERSION:
        raise ValueError('Reevaluate all six baselines with the current shared extractor first')
    if (fingerprint(profiles) != baseline['settings']['profile_sha256'] or
            fingerprint(refs) != baseline['settings']['reference_sha256']):
        raise ValueError('Pilot profiles or references changed')
    predictions = read(PILOT / 'predictions.json')
    if fingerprint(predictions) != baseline['settings']['predictions_sha256']:
        raise ValueError('Frozen pilot predictions changed')
    # Require the same versioned extractor for all saved methods and PaddleOCR.
    for method in FAMILIES:
        for doc in SOURCES:
            text = field_text('\n\n'.join(data[doc][method]['pages']), 'markdown')
            if extract(doc, text) != predictions['methods'][method]['documents'][doc]['fields']:
                raise ValueError('Shared extractor does not reproduce refreshed baseline predictions')
    manifest_path = ocr_root / 'manifest.json'
    manifest = read(manifest_path)
    if manifest['failures'] or len(manifest['documents']) != len(SOURCES):
        raise ValueError('Require a complete three-document PaddleOCR run')
    paddle_docs, paddle_pages, timing, metadata_records = {}, {}, [], {}
    for doc, filename in SOURCES.items():
        records = [r for r in manifest['documents'] if r['source'] == filename]
        if len(records) != 1:
            raise ValueError(f'{doc}: require exactly one selected run')
        record = records[0]
        if record['source_sha256'] != SOURCE_SHA256[doc]:
            raise ValueError('PaddleOCR source differs from the frozen pilot source')
        responses, metadata = verify_run(record, ROOT / 'data' / filename, ocr_root, PAGE_COUNTS[doc])
        paddle_pages[doc] = [r['text'] for r in responses]
        metadata_records[doc] = metadata
        raw_text = '\n\n'.join(r['text'] for r in responses)
        data[doc][METHOD] = {'pages': [raw_text], 'input_format': 'plain'}
        text = field_text(raw_text, 'plain')
        decoded = output / 'decoded' / METHOD
        decoded.mkdir(parents=True, exist_ok=True)
        (decoded / (doc + '.txt')).write_text(text, encoding='utf-8')
        fields = extract(doc, text)
        for item in fields:
            a, b = item['decoded_span']
            if item['value'] != text[a:b]:
                raise ValueError('Extractor changed a captured OCR value')
        paddle_docs[doc] = {'fields': fields}
        for r in responses:
            timing.append({'document': doc, 'page': r['page'], 'seconds': r['seconds'],
                           'recognized_lines': len(r['raw']['rec_texts']), 'characters': len(r['text']),
                           'orientation_degrees': r['raw'].get('doc_preprocessor_res', {}).get('angle')})
    predictions['methods'][METHOD] = {
        'family': 'paddleocr', 'documents': paddle_docs,
        'provenance': {'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
                       'metadata': metadata_records,
                       'extractor_version': EXTRACTOR_VERSION,
                       'extraction': 'Shared formatting-tolerant section/label rules; native plain text; raw spans; no value repair.'},
    }
    configs = [{k: v for k, v in metadata.items() if k not in ('source', 'responses')}
               for metadata in metadata_records.values()]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError('PaddleOCR documents use different model/runtime/settings configurations')
    result = evaluate_benchmark(profiles, refs, predictions, split='validation')
    for method in FAMILIES:
        if (result['methods'][method]['score'] != baseline['methods'][method]['score'] or
                result['methods'][method]['metrics'] != baseline['methods'][method]['metrics']):
            raise ValueError('Baseline pilot scores changed; investigate before comparing')
    result['meaning'] = baseline['meaning']
    result['extractor_version'] = EXTRACTOR_VERSION
    result['pilot_limitations'] = baseline['pilot_limitations']
    rows, audit = [], []
    for rank in result['ranking']:
        method = rank['method']
        scored = result['methods'][method]
        rows.append({**rank, 'correct': scored['metrics']['critical_correct'], 'total': scored['metrics']['critical_fields'],
                     **{d + '_correct': r['counts']['critical_correct'] for d, r in scored['documents'].items()},
                     'missing': scored['metrics']['missing_fields'], 'wrong': scored['metrics']['wrong_values']})
        for doc, r in scored['documents'].items():
            for f in r['field_results']:
                literal_present = ''
                if method == METHOD:
                    page_text = paddle_pages[doc][f['reference_location']['page'] - 1]
                    literal_present = ' '.join(str(f['reference_value']).split()) in ' '.join(page_text.split())
                audit.append({'method': method, 'document': doc, 'field': f['field'],
                              'source_page': f['reference_location']['page'], 'reference': f['reference_value'],
                              'prediction': f['predicted_value'], 'result': f['reason'],
                              'reference_literal_present_on_ocr_page': literal_present})
    for name, value in [('profiles.json', profiles), ('references.json', refs), ('predictions.json', predictions),
                        ('critical_fields.json', result), ('input.json', data)]:
        write_json(output / name, value)
    write_csv(output / 'critical_field_ranking.csv', rows)
    write_csv(output / 'field_audit.csv', audit)
    write_csv(output / 'page_timing.csv', timing)
    print('Critical-field results:', json.dumps(rows, indent=2), flush=True)

    families = {**FAMILIES, METHOD: 'paddleocr'}
    comparisons = {}
    for mode, ontology in [('with_ontology', ROOT / 'document_ontology.json'), ('without_ontology', None)]:
        print(f'Evaluating seven whole-document outputs: {mode}', flush=True)
        consensus, pairs, volumes, normalized = evaluate(data, ontology_path=ontology, weighting='document', overlap_weight=1.0)
        consensus['comparison_unit'] = 'whole_document; page=1 is a matrix index, not a physical page'
        sensitivity = consensus_family_sensitivity(pairs, families)
        comparisons[mode] = {'ranking': consensus['ranking'],
                             'family_balanced_scores': sensitivity['family_balanced_scores']}
        folder = output / mode
        folder.mkdir(exist_ok=True)
        for name, value in [('results.json', consensus), ('pair_differences.json', pairs),
                            ('text_volume.json', volumes), ('normalized.json', normalized),
                            ('family_sensitivity.json', sensitivity)]:
            write_json(folder / name, value)
        write_csv(folder / 'ranking.csv', consensus['ranking'])
    write_json(output / 'run_manifest.json', {
        'ocr_manifest': str(manifest_path), 'ocr_manifest_sha256': sha(manifest_path),
        'baseline_input_sha256': sha(SAVED / 'input.json'), 'input_sha256': sha(output / 'input.json'),
        'pilot_predictions_sha256': sha(PILOT / 'predictions.json'),
        'evaluation_script_sha256': sha(Path(__file__)),
        'extractor_script_sha256': sha(ROOT / 'scripts/evaluate_six_critical_fields.py'),
        'value_normalization_script_sha256': sha(ROOT / 'standalone_ocr_eval/ocr_eval/field_values.py'),
        'extractor_version': EXTRACTOR_VERSION,
        'source_sha256': SOURCE_SHA256, 'verified_baseline_artifacts': verified,
        'baseline_provenance': provenance, 'families': families,
    })
    changes = compare_previous_run(output, result, profiles, refs, comparisons)
    make_report(output, rows, audit, timing, comparisons, configs[0], changes)
    print(output / 'report.md', flush=True)


def compare_previous_run(output, current, profiles, refs, comparisons):
    """Separate extraction changes from OCR changes using the archived text."""
    archive = ROOT / 'output/extractor_v1_orientation_enabled_archive/paddleocr_v4_experiment'
    if not (archive / 'critical_fields.json').exists():
        return None
    previous = read(archive / 'critical_fields.json')
    if (fingerprint(profiles) != previous['settings']['profile_sha256'] or
            fingerprint(refs) != previous['settings']['reference_sha256']):
        raise ValueError('Previous and current scores use different references or profiles')
    data = read(archive / 'input.json')
    predictions = read(archive / 'predictions.json')
    for method, predicted in predictions['methods'].items():
        for doc, records in data.items():
            raw = records[method]
            # The historical pilot decoded all six baselines as Markdown;
            # preserve that convention in this extraction-only ablation too.
            fmt = 'markdown' if method in FAMILIES else raw.get('input_format', 'plain')
            text = field_text('\n\n'.join(raw['pages']), fmt)
            predicted['documents'][doc]['fields'] = extract(doc, text)
    extractor_only = evaluate_benchmark(profiles, refs, predictions, split='validation')
    rows = [{'method': method, 'previous_score': old['score'],
             'extractor_only_score': extractor_only['methods'][method]['score'],
             'current_score': current['methods'][method]['score'],
             'previous_correct': old['metrics']['critical_correct'],
             'current_correct': current['methods'][method]['metrics']['critical_correct']}
            for method, old in previous['methods'].items()]
    changes = {'archive': str(archive), 'extractor_version': EXTRACTOR_VERSION,
               'critical_fields': rows, 'consensus': {}}
    for mode, new in comparisons.items():
        old = read(archive / mode / 'results.json')
        changes['consensus'][mode] = {
            'previous': old['ranking'], 'current': new['ranking']}
    write_json(output / 'change_comparison.json', changes)
    return changes


def make_report(output, rows, audit, timing, comparisons, config, changes=None):
    paddle = next(r for r in rows if r['method'] == METHOD)
    seconds = sum(r['seconds'] for r in timing)
    options, versions = config['options'], config['runtime_versions']
    lines = ['# PaddleOCR v4 additional experiment', '',
             f'Processed all 48 pages in the three existing PDFs locally. PP-OCRv4 passes {paddle["correct"]}/27 '
             f'selected critical fields, with an equal-document score of {paddle["critical_field_score"]:.2f}/100.', '',
             '## Configuration', '',
             f'- English PP-OCRv4: `PP-OCRv4_mobile_det` + `en_PP-OCRv4_mobile_rec`; PaddleOCR {versions["paddleocr"]}, PaddlePaddle {versions["paddlepaddle"]}, PaddleX {versions["paddlex"]}.',
             f'- CPU, {options["cpu_threads"]} threads; PDFs rendered at {config["dpi"]} DPI, detector maximum side {options["text_det_limit_side_len"]} pixels; recognition confidence threshold {options["text_rec_score_thresh"]}.',
             '- Document orientation uses PP-LCNet_x1_0; text-line orientation is '
             f'{"enabled" if options["use_textline_orientation"] else "disabled"}. '
             'Document unwarping is disabled. Auxiliary orientation models are separate from the v4 detector/recognizer.',
             '- Native recognition order, one detection per line. No table reconstruction, field-specific corrections, or tuning against the pilot answers.',
             '- Raw JSON retains confidence scores, polygons and rotation metadata. Model files, source PDFs, settings and page artifacts have SHA-256 provenance.', '',
             '## Frozen critical-field pilot', '',
             f'The same 27 references, profiles, and shared extractor version {EXTRACTOR_VERSION} are used for all seven outputs. '
             'All six saved methods are re-extracted with the current rules, then hash-verified and reproduced before adding PaddleOCR. '
             'Whitespace, token punctuation and schedule anchoring are tolerated; captured OCR values remain uncorrected. '
             'Typed month dates accept either spacing after the comma; literal comparison fields remain literal. '
             'Scores average fields within each document, then weight the three documents equally.', '',
             '| Method | Score / 100 | Correct / 27 | Board / 13 | CSA / 6 | Trade / 8 | Unextracted | Wrong value/binding |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["method"]} | {r["critical_field_score"]:.2f} | {r["correct"]} | '
                     f'{r["board_correct"]} | {r["csa_correct"]} | {r["trade_correct"]} | {r["missing"]} | {r["wrong"]} |')
    if changes:
        lines += ['', '## Change from the previous run', '',
                  'The middle column applies the current shared extractor to the previous OCR text. '
                  'The last column also uses the new PaddleOCR run with text-line orientation disabled. '
                  'References and field weights are identical. This post-hoc comparison is exploratory.', '',
                  '| Method | Previous score | Extractor only | Current score | Previous correct | Current correct |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in changes['critical_fields']:
            lines.append(f'| {r["method"]} | {r["previous_score"]:.2f} | {r["extractor_only_score"]:.2f} | '
                         f'{r["current_score"]:.2f} | {r["previous_correct"]} | {r["current_correct"]} |')
        lines += ['', 'The previous reports, field predictions and manifest are preserved under '
                  '`output/extractor_v1_orientation_enabled_archive/`; original OCR checkpoints remain in their '
                  'settings-specific run directories. `change_comparison.json` includes both consensus rankings.', '']
    lines += ['', 'PaddleOCR failures under the shared extractor:', '',
              '| Document | Page | Field | Reference | Extracted | Result | Reference literal on OCR page |',
              '|---|---:|---|---|---|---|---|']
    for r in audit:
        if r['method'] == METHOD and r['result'] != 'correct':
            def cell(x):
                return str(x).replace('|', '\\|').replace('\n', ' ') if x is not None else '—'
            lines.append('| ' + ' | '.join(cell(r[k]) for k in ['document', 'source_page', 'field', 'reference', 'prediction', 'result', 'reference_literal_present_on_ocr_page']) + ' |')
    lines += ['', 'Unextracted means the shared parser did not bind a value. It does not establish that the OCR omitted '
              'the value. Inspect `decoded/paddleocr-v4/` and `field_audit.csv` to distinguish label/order loss from transcription errors.', '',
              'The literal-presence column searches only the corresponding physical OCR page, after collapsing whitespace. '
              'It is a diagnostic, awards no credit, and does not establish correct field binding. '
              'A false value may reflect transcription or formatting differences rather than omission.', '',
              '## Whole-document consensus', '',
              'All seven outputs are compared as whole documents because Insavlo has no reliable page boundaries. '
              'Each document has equal weight. These scores measure agreement with peers, not correctness. '
              'Adding a seventh method changes every peer average.', '',
              '| Method | Ontology-region agreement | Family-balanced agreement | Whole-text token agreement |',
              '|---|---:|---:|---:|']
    tokens = {r['model']: r['consensus_score'] for r in comparisons['without_ontology']['ranking']}
    for r in comparisons['with_ontology']['ranking']:
        m = r['model']
        lines.append(f'| {m} | {r["consensus_score"]:.2f} | {comparisons["with_ontology"]["family_balanced_scores"][m]:.2f} | {tokens[m]:.2f} |')
    lines += ['', '## Inference timing', '',
              f'Total page inference: {seconds:.2f} seconds ({seconds/60:.2f} minutes), mean {seconds/48:.2f} seconds/page. '
              'Excludes package/model downloads, model loading, PDF rendering, and evaluation. '
              'Includes orientation preprocessing, detection and recognition. '
              'No matched hardware timing exists for the saved peers, so this is not a speed ranking.', '',
              '| Document | Pages | Inference seconds | Seconds/page |', '|---|---:|---:|---:|']
    for doc, count in PAGE_COUNTS.items():
        duration = sum(r['seconds'] for r in timing if r['document'] == doc)
        lines.append(f'| {doc} | {count} | {duration:.2f} | {duration/count:.2f} |')
    lines += ['', '## Limits and provenance', '',
              '- This is an exploratory OCR-plus-extraction pilot on seven selected source pages, not held-out accuracy or pure OCR CER. References were image-checked by Codex, not independently human-adjudicated. No new reference labels were introduced.',
              '- Shared extraction rules tolerate formatting but remain sensitive to damaged labels, reading order and tables. Native line output can score poorly even when isolated values are present. Localization is not scored.',
              '- Extractor version 2 addresses observed failures in this corpus. Scores are post-hoc exploratory results, not held-out accuracy; reference values and comparison policies are unchanged.',
              '- The saved openai-6 and fable-5-1 labels refer to Tesseract-based artifacts, including differing review workflows. They are grouped as one family. Insavlo engine independence is unknown.',
              '- Fable and Insavlo lack verified source hashes. Mistral trade used an earlier TIFF with unverified pixel equivalence. DeepSeek trade page 10 was truncated. See baseline provenance in run_manifest.json.',
              '- Model downloads came from official Paddle model hosting. Document inference ran locally in myenv3.13; outputs contain uncorrected OCR errors.', '',
              '## Reproduce', '', '```bash', 'bash scripts/setup_paddleocr.sh',
              'conda run --no-capture-output -n myenv3.13 python scripts/evaluate_six_critical_fields.py',
              'conda run --no-capture-output -n myenv3.13 python scripts/run_paddleocr_v4.py --no-textline-orientation',
              'conda run --no-capture-output -n myenv3.13 python scripts/evaluate_paddleocr_v4.py', '```', '',
              'Raw runs: `output/paddleocr-v4/`. This report directory includes frozen input, predictions, '
              'field audit, page timing, pair differences, consensus rankings and family sensitivity.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
