"""Compare gateway-reported Claude OCR with verified frozen outputs, without ontology."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from ocr_common import ROOT, write_json
from run_claude_ocr import MODEL, BASE_URL, validate_checkpoint
from evaluate_six_critical_fields import EXTRACTOR_VERSION
from evaluate_openai_vlm import (
    BASE as PADDLE_BASE, EXCLUDED_METHODS as BASE_EXCLUDED_METHODS, verify_openai, read, write_csv,
    verify_run, PAGE_COUNTS, SOURCES, SOURCE_SHA256, extract, sha,
    evaluate, evaluate_benchmark, consensus_family_sensitivity, field_text, fingerprint,
)

BASE = ROOT / 'output/openai_vlm_experiment'
EXCLUDED_METHODS = BASE_EXCLUDED_METHODS | {'fable-5-1'}


def verified_baseline():
    manifest = read(BASE / 'run_manifest.json')
    if sha(PADDLE_BASE / 'run_manifest.json') != manifest['baseline_manifest_sha256']:
        raise ValueError('Paddle baseline manifest changed')
    frozen = read(PADDLE_BASE / 'run_manifest.json')
    if sha(PADDLE_BASE / 'input.json') != frozen['input_sha256']:
        raise ValueError('Frozen baseline input changed')
    for path, expected in frozen['verified_baseline_artifacts'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Frozen baseline artifact changed')
    if sha(ROOT / 'scripts/evaluate_six_critical_fields.py') != frozen['extractor_script_sha256']:
        raise ValueError('Frozen extractor changed')
    paddle_path = Path(frozen['ocr_manifest'])
    if sha(paddle_path) != frozen['ocr_manifest_sha256']:
        raise ValueError('Paddle manifest changed')
    paddle = read(paddle_path)
    for doc, source in SOURCES.items():
        record, = [r for r in paddle['documents'] if r['source'] == source]
        verify_run(record, ROOT / 'data' / source, paddle_path.parent, PAGE_COUNTS[doc])
    expected_data = read(PADDLE_BASE / 'input.json')
    for methods in expected_data.values():
        for method in BASE_EXCLUDED_METHODS:
            methods.pop(method, None)
    openai_path = Path(manifest['ocr_manifest'])
    if sha(openai_path) != manifest['ocr_manifest_sha256']:
        raise ValueError('OpenAI manifest changed')
    openai = read(openai_path)
    if openai['failures'] or len(openai['documents']) != 3:
        raise ValueError('Incomplete OpenAI baseline')
    for doc, source in SOURCES.items():
        record, = [r for r in openai['documents'] if r['source'] == source]
        responses, config = verify_openai(record, openai_path.parent, doc)
        expected_data[doc][config['model']] = {
            'pages': ['\n\n'.join(r['markdown'] for r in responses)], 'input_format': 'markdown'}
    data = read(BASE / 'input.json')
    if data != expected_data:
        raise ValueError('Baseline comparison input differs from verified source artifacts')
    profiles, refs, predictions = [read(BASE / n) for n in ('profiles.json', 'references.json', 'predictions.json')]
    frozen_fields = read(PADDLE_BASE / 'critical_fields.json')
    fields = read(BASE / 'critical_fields.json')
    for key, value in [('profile_sha256', profiles), ('reference_sha256', refs)]:
        if fingerprint(value) != frozen_fields['settings'][key]:
            raise ValueError('Frozen pilot reference or profile changed')
    if fingerprint(predictions) != fields['settings']['predictions_sha256']:
        raise ValueError('Baseline field predictions changed')
    for doc, methods in data.items():
        for method, record in methods.items():
            # Historical pilot decoded all original outputs as Markdown; preserve that convention.
            fmt = 'plain' if method == 'paddleocr-v4' else 'markdown'
            actual = extract(doc, field_text('\n\n'.join(record['pages']), fmt))
            if predictions['methods'][method]['documents'][doc]['fields'] != actual:
                raise ValueError('Baseline predictions differ from frozen extraction')
    return data, profiles, refs, predictions, fields, manifest


def verify_claude(record, root, document):
    source = ROOT / 'data' / SOURCES[document]
    if record['source_sha256'] != sha(source) or sha(source) != SOURCE_SHA256[document]:
        raise ValueError('Source PDF hash mismatch')
    folder = (root / record['run']).resolve()
    if not folder.is_relative_to(root.resolve()):
        raise ValueError('Run path escapes output root')
    if sha(folder / 'source.json') != record['source_metadata_sha256']:
        raise ValueError('Source metadata hash mismatch')
    metadata = read(folder / 'source.json')
    if metadata['model'] != MODEL or metadata['api_base'] != BASE_URL:
        raise ValueError('Unexpected requested model or gateway')
    if metadata['source_sha256'] != record['source_sha256'] or metadata['source'] != record['source']:
        raise ValueError('Source metadata mismatch')
    config = {k: v for k, v in metadata.items() if k not in ('source', 'source_sha256', 'responses', 'artifact_sha256')}
    digest = hashlib.sha256(source.read_bytes())
    digest.update(json.dumps(config, sort_keys=True).encode())
    if folder.name != digest.hexdigest()[:16]:
        raise ValueError('Source/settings cache hash mismatch')
    paths = sorted(folder.glob('response-*.json'))
    expected = PAGE_COUNTS[document]
    if len(paths) != expected or metadata['responses'] != expected or record['pages'] != expected:
        raise ValueError('Incomplete or extra page responses')
    responses = []
    for index, path in enumerate(paths, 1):
        if path.name != f'response-{index:04d}.json' or sha(path) != metadata['artifact_sha256'].get(path.name):
            raise ValueError('Checkpoint hash or order mismatch')
        response = read(path)
        validate_checkpoint(response, index, MODEL)
        responses.append(response)
    return responses, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ocr-output', type=Path, default=ROOT / 'output/claude-vlm-fable-5-1')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/claude_vlm_experiment')
    parser.add_argument('--complete-documents-only', action='store_true',
                        help='Explicitly compare only documents completed by Claude, filtering every peer identically')
    parser.add_argument('--rescore-existing', action='store_true',
                        help='Apply method exclusions to verified saved comparison inputs without rerunning extraction')
    args = parser.parse_args()
    root, out = args.ocr_output.resolve(), args.output.resolve()
    if args.rescore_existing:
        rescore_existing(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    data, profiles, refs, predictions, baseline, baseline_manifest = verified_baseline()
    for methods in data.values():
        for method in EXCLUDED_METHODS:
            methods.pop(method, None)
    for method in EXCLUDED_METHODS:
        predictions['methods'].pop(method, None)
    manifest_path = root / 'manifest.json'
    manifest = read(manifest_path)
    if manifest['model'] != MODEL or manifest['api_base'] != BASE_URL:
        raise ValueError('Unexpected model or endpoint')
    if manifest['failures'] or len(manifest['documents']) != 3:
        if not args.complete_documents_only or not manifest['documents']:
            raise ValueError('Require a complete run, or explicitly select --complete-documents-only')
    selected = {d: f for d, f in SOURCES.items() if any(r['source'] == f for r in manifest['documents'])}
    if len(selected) != len(manifest['documents']):
        raise ValueError('Duplicate or unknown completed documents')
    data = {d: data[d] for d in selected}
    refs['documents'] = {d: refs['documents'][d] for d in selected}
    for method in predictions['methods'].values():
        method['documents'] = {d: method['documents'][d] for d in selected}
    expected_fields = evaluate_benchmark(profiles, refs, predictions, split='validation')
    documents, configs, timing, provenance = {}, [], [], {}
    for doc, filename in selected.items():
        record, = [r for r in manifest['documents'] if r['source'] == filename]
        responses, config = verify_claude(record, root, doc)
        configs.append(config)
        provenance[doc] = record
        markdown = '\n\n'.join(r['markdown'] for r in responses)
        data[doc][MODEL] = {'pages': [markdown], 'input_format': 'markdown'}
        text = field_text(markdown, 'markdown')
        (out / (doc + '-decoded.txt')).write_text(text, encoding='utf-8')
        documents[doc] = {'fields': extract(doc, text)}
        for r in responses:
            raw = r['raw_response']; usage = raw.get('usage') or {}
            timing.append({'document': doc, 'page': r['page_number'], 'seconds': r['seconds'],
                           'returned_model': raw['model'], 'response_id': raw['id'],
                           'input_tokens': usage.get('prompt_tokens', 0),
                           'output_tokens': usage.get('completion_tokens', 0)})
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError('Claude documents use different configurations')
    predictions['methods'][MODEL] = {'family': 'claude-gateway-reported', 'documents': documents,
                                    'provenance': {'manifest': str(manifest_path), 'sha256': sha(manifest_path),
                                                   'config': configs[0], 'runs': provenance}}
    fields = evaluate_benchmark(profiles, refs, predictions, split='validation')
    fields['meaning'], fields['pilot_limitations'] = baseline['meaning'], baseline['pilot_limitations']
    fields['extractor_version'] = EXTRACTOR_VERSION
    for method, previous in expected_fields['methods'].items():
        if fields['methods'][method]['score'] != previous['score']:
            raise ValueError('Baseline pilot score changed')
    audit = [{'method': method, 'document': doc, 'field': f['field'],
              'source_page': f['reference_location']['page'], 'reference': f['reference_value'],
              'prediction': f['predicted_value'], 'result': f['reason']}
             for method, scored in fields['methods'].items() for doc, result in scored['documents'].items()
             for f in result['field_results']]
    write_csv(out / 'field_audit.csv', audit)
    write_csv(out / 'page_timing.csv', timing)
    write_csv(out / 'critical_field_ranking.csv', fields['ranking'])
    print('Critical fields:', json.dumps(fields['ranking']), flush=True)
    print('Computing whole-text comparison without ontology...', flush=True)
    result, pairs, volumes, normalized = evaluate(data, ontology_path=None, overlap_weight=1.0, weighting='document')
    result['comparison_unit'] = 'whole_document; page=1 is a matrix index, not a physical page'
    result['excluded_methods'] = sorted(EXCLUDED_METHODS)
    result['included_documents'] = list(selected)
    result['excluded_incomplete_documents'] = sorted(set(SOURCES) - set(selected))
    families = {m: f for m, f in baseline_manifest['families'].items() if m not in EXCLUDED_METHODS}
    families[MODEL] = 'claude-gateway-reported'
    sensitivity = consensus_family_sensitivity(pairs, families)
    for name, value in [('input.json', data), ('profiles.json', profiles), ('references.json', refs),
                        ('predictions.json', predictions), ('critical_fields.json', fields),
                        ('results.json', result), ('pair_differences.json', pairs),
                        ('text_volume.json', volumes), ('normalized.json', normalized),
                        ('family_sensitivity.json', sensitivity)]:
        write_json(out / name, value)
    write_csv(out / 'ranking.csv', result['ranking'])
    write_json(out / 'run_manifest.json', {'ocr_manifest': str(manifest_path), 'ocr_manifest_sha256': sha(manifest_path),
               'baseline_manifest': str(BASE / 'run_manifest.json'), 'baseline_manifest_sha256': sha(BASE / 'run_manifest.json'),
               'baseline_input_sha256': sha(BASE / 'input.json'), 'input_sha256': sha(out / 'input.json'),
               'source_sha256': SOURCE_SHA256, 'evaluation_script_sha256': sha(Path(__file__)),
               'runner_script_sha256': sha(ROOT / 'scripts/run_claude_ocr.py'), 'families': families,
               'excluded_methods': sorted(EXCLUDED_METHODS), 'included_documents': list(selected),
               'excluded_incomplete_documents': sorted(set(SOURCES) - set(selected))})
    make_report(out, result, fields, timing, manifest, configs[0])
    print(out / 'report.md', flush=True)


def rescore_existing(out):
    manifest = read(out / 'run_manifest.json')
    if sha(out / 'input.json') != manifest['input_sha256']:
        raise ValueError('Saved comparison input changed')
    data = read(out / 'input.json')
    profiles, refs, predictions = [read(out / n) for n in ('profiles.json', 'references.json', 'predictions.json')]
    previous = read(out / 'critical_fields.json')
    for key, value in [('profile_sha256', profiles), ('reference_sha256', refs), ('predictions_sha256', predictions)]:
        if fingerprint(value) != previous['settings'][key]:
            raise ValueError('Saved pilot inputs changed')
    ocr_path = Path(manifest['ocr_manifest'])
    if sha(ocr_path) != manifest['ocr_manifest_sha256']:
        raise ValueError('OCR manifest changed')
    ocr = read(ocr_path)
    for doc in data:
        record, = [r for r in ocr['documents'] if r['source'] == SOURCES[doc]]
        responses, config = verify_claude(record, ocr_path.parent, doc)
        if data[doc][MODEL]['pages'] != ['\n\n'.join(r['markdown'] for r in responses)]:
            raise ValueError('Saved Claude input differs from raw responses')
        for method in EXCLUDED_METHODS:
            data[doc].pop(method, None)
    for method in EXCLUDED_METHODS:
        predictions['methods'].pop(method, None)
    fields = evaluate_benchmark(profiles, refs, predictions, split='validation')
    for method in fields['methods']:
        if fields['methods'][method]['score'] != previous['methods'][method]['score']:
            raise ValueError('Saved field scores changed')
    fields['meaning'], fields['pilot_limitations'] = previous['meaning'], previous['pilot_limitations']
    result, pairs, volumes, normalized = evaluate(data, ontology_path=None, overlap_weight=1.0, weighting='document')
    result.update(comparison_unit='whole_document; page=1 is a matrix index, not a physical page',
                  excluded_methods=sorted(EXCLUDED_METHODS), included_documents=list(data),
                  excluded_incomplete_documents=manifest['excluded_incomplete_documents'])
    families = {m: f for m, f in manifest['families'].items() if m not in EXCLUDED_METHODS}
    sensitivity = consensus_family_sensitivity(pairs, families)
    with (out / 'field_audit.csv').open() as stream:
        audit = [r for r in csv.DictReader(stream) if r['method'] not in EXCLUDED_METHODS]
    with (out / 'page_timing.csv').open() as stream:
        timing = list(csv.DictReader(stream))
    for row in timing:
        for key in ('seconds', 'input_tokens', 'output_tokens'):
            row[key] = float(row[key]) if key == 'seconds' else int(row[key])
    write_json(out / 'prior_run_manifest.json', manifest)
    for name, value in [('input.json', data), ('predictions.json', predictions), ('critical_fields.json', fields),
                        ('results.json', result), ('pair_differences.json', pairs), ('text_volume.json', volumes),
                        ('normalized.json', normalized), ('family_sensitivity.json', sensitivity)]:
        write_json(out / name, value)
    write_csv(out / 'ranking.csv', result['ranking'])
    write_csv(out / 'critical_field_ranking.csv', fields['ranking'])
    write_csv(out / 'field_audit.csv', audit)
    manifest.update(input_sha256=sha(out / 'input.json'), families=families,
                    excluded_methods=sorted(EXCLUDED_METHODS), evaluation_script_sha256=sha(Path(__file__)),
                    rescore_note='Method exclusions applied to saved verified inputs; field extraction was not rerun.')
    write_json(out / 'run_manifest.json', manifest)
    make_report(out, result, fields, timing, ocr, config)
    print(json.dumps(result['ranking'], indent=2))


def make_report(out, result, fields, timing, manifest, config):
    docs = list(result['included_documents'])
    page_count = sum(PAGE_COUNTS[d] for d in docs)
    field_counts = {'board': 13, 'csa': 6, 'trade': 8}
    total_fields = sum(field_counts[d] for d in docs)
    lines = ['# Claude OCR experiment — comparison without ontology', '',
             f'The experiment used `{BASE_URL}` with `{MODEL}`. '
             f'This comparison includes {page_count} pages across {len(docs)} complete documents: {", ".join(docs)}. '
             'Every included response reports that model and a completed (`stop`) finish reason. '
             'Model identity is gateway-reported; upstream inference is not independently verified.', '',
             f'Same PDFs, {config["dpi"]} DPI, high image detail, transcription system prompt, and 16,000-token output limit '
             'as the OpenAI VLM experiment. This gateway uses Chat Completions with JSON-object mode and the schema '
             'in the user message because it did not enforce strict JSON-schema mode in a synthetic test. '
             'Reasoning settings are provider defaults. No benchmark answers or other OCR outputs were supplied.', '',
             '## Whole-document comparison', '',
             f'Token overlap only, without ontology or sequence weighting. Scores average agreement with {len(result["models"]) - 1} peers '
             'and weight the included documents equally. They measure peer consensus, not accuracy. Adding a method '
             'changes all consensus scores. Historical `openai-6` and `fable-5-1` outputs are excluded.', '',
             '| Rank | Output | Overall / 100 | ' + ' | '.join(docs) + ' |',
             '|---:|---|---:|' + '---:|' * len(docs)]
    for row in result['ranking']:
        scores = [next(x['consensus_score'] for x in result['per_document_rankings'][d] if x['model'] == row['model'])
                  for d in docs]
        lines.append(f'| {row["rank"]} | {row["model"]} | {row["consensus_score"]:.2f} | ' +
                     ' | '.join(f'{s:.2f}' for s in scores) + ' |')
    lines += ['', '## Frozen critical-field pilot', '',
              f'Same {total_fields} selected fields for the included documents, image-checked references, and shared extractor version {EXTRACTOR_VERSION}. '
              'The rules tolerate formatting and preserve raw OCR spans; references and comparison policies are unchanged. '
              'This exploratory OCR-plus-extraction pilot is not independently adjudicated or held-out accuracy.', '',
              f'| Output | Score / 100 | Correct / {total_fields} | ' + ' | '.join(f'{d} / {field_counts[d]}' for d in docs) + ' |',
              '|---|---:|---:|' + '---:|' * len(docs)]
    for row in fields['ranking']:
        r = fields['methods'][row['method']]
        counts = [r['documents'][d]['counts']['critical_correct'] for d in docs]
        lines.append(f'| {row["method"]} | {row["critical_field_score"]:.2f} | {r["metrics"]["critical_correct"]} | ' +
                     ' | '.join(map(str, counts)) + ' |')
    lines += ['', '## Runtime and provenance', '',
              f'- Wall time: {manifest["wall_seconds"]:.2f} seconds, {manifest["workers"]} concurrent document workers. '
              + manifest.get('wall_time_scope', 'Includes requests/retries, rendering and saving; resumed runs can reuse checkpoints.'),
              f'- Included documents only, summed page request time: {sum(r["seconds"] for r in timing):.2f} seconds; concurrent requests overlap.',
              f'- Included documents only, gateway-reported usage: {sum(r["input_tokens"] for r in timing):,} prompt tokens and '
              f'{sum(r["output_tokens"] for r in timing):,} completion tokens. Billing is not independently verified.',
              '- Raw responses, source/settings hashes, page image hashes, response IDs and requested/returned model IDs are retained. No API key is saved.',
              '- Historical Tesseract-based `fable-5-1` artifacts remain on disk but are excluded; `claude-fable-5-1` is the new gateway run.',
              '- Existing limitations: Insavlo engine/source provenance unverified; historical Fable source hashes unverified; '
              'Mistral trade used an earlier TIFF; DeepSeek trade page 10 was truncated.', '',
              '## Reproduce', '', 'Set OPENAI_API_KEY locally, then:', '', '```bash',
              'conda run --no-capture-output -n myenv3.13 python scripts/run_claude_ocr.py --json-mode json_object --workers 3',
              'conda run --no-capture-output -n myenv3.13 python scripts/evaluate_claude_vlm.py' +
              (' --complete-documents-only' if manifest['failures'] else ''), '```', '',
              'Evaluation uses saved outputs only. See field_audit.csv, page_timing.csv, ranking.csv and run_manifest.json.', '']
    if manifest['failures']:
        failures = ['', '## Incomplete documents excluded from every method', '',
                    'The full 48-page comparison is unavailable. The scores below use an identical subset for every method '
                    'and cannot be compared directly with the earlier three-document scores.', '']
        for failure in manifest['failures']:
            failures.append(f'- {failure["source"]}: {failure.get("completed_pages", "unknown")} pages completed.')
            for page in failure.get('failed_pages', []):
                failures.append(f'  - Page {page["page_number"]}: HTTP {page.get("http_status")}; {page.get("message")}.')
        failures.append('')
        lines[4:4] = failures
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
