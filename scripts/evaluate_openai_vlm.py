"""Add verified GPT-6 Astra page responses to the frozen seven-output experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from ocr_common import ROOT, write_json
from run_astra_ocr import MODEL, validate_checkpoint
from evaluate_paddleocr_v4 import read, write_csv, verify_run, PAGE_COUNTS
from evaluate_six_critical_fields import SOURCES, SOURCE_SHA256, extract, sha, EXTRACTOR_VERSION

sys.path.insert(0, str(ROOT / 'standalone_ocr_eval'))
from ocr_eval import evaluate, evaluate_benchmark, consensus_family_sensitivity
from ocr_eval.field_values import field_text
from ocr_eval.profiles import fingerprint

BASE = ROOT / 'output/paddleocr_v4_experiment'
EXCLUDED_METHODS = {'openai-6'}


def verify_openai(record, root, document):
    source = ROOT / 'data' / SOURCES[document]
    if record['source_sha256'] != sha(source) or sha(source) != SOURCE_SHA256[document]:
        raise ValueError('Source PDF hash mismatch')
    folder = (root / record['run']).resolve()
    if not folder.is_relative_to(root.resolve()):
        raise ValueError('Run path escapes output root')
    if sha(folder / 'source.json') != record['source_metadata_sha256']:
        raise ValueError('Source metadata hash mismatch')
    metadata = read(folder / 'source.json')
    if metadata['model'] != MODEL or metadata['api_base'] != 'https://api.openai.com/v1':
        raise ValueError('Unexpected requested model or API endpoint')
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
        raise ValueError('Incomplete or extra OCR page responses')
    responses = []
    for index, path in enumerate(paths, 1):
        if path.name != f'response-{index:04d}.json' or sha(path) != metadata['artifact_sha256'].get(path.name):
            raise ValueError('Checkpoint hash or order mismatch')
        response = read(path)
        validate_checkpoint(response, index)
        responses.append(response)
    return responses, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ocr-output', type=Path, default=ROOT / 'output/openai-vlm-gpt-6-astra')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/openai_vlm_experiment')
    args = parser.parse_args()
    root, out = args.ocr_output.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    baseline_manifest = read(BASE / 'run_manifest.json')
    if sha(BASE / 'input.json') != baseline_manifest['input_sha256']:
        raise ValueError('Frozen seven-method input changed')
    for path, expected in baseline_manifest['verified_baseline_artifacts'].items():
        if sha(Path(path)) != expected:
            raise ValueError(f'Baseline artifact changed: {path}')
    paddle_manifest_path = Path(baseline_manifest['ocr_manifest'])
    if sha(paddle_manifest_path) != baseline_manifest['ocr_manifest_sha256']:
        raise ValueError('PaddleOCR manifest changed')
    paddle_manifest = read(paddle_manifest_path)
    for doc, source in SOURCES.items():
        record, = [r for r in paddle_manifest['documents'] if r['source'] == source]
        verify_run(record, ROOT / 'data' / source, paddle_manifest_path.parent, PAGE_COUNTS[doc])
    if sha(ROOT / 'scripts/evaluate_six_critical_fields.py') != baseline_manifest['extractor_script_sha256']:
        raise ValueError('Frozen field extraction rules changed')
    data = read(BASE / 'input.json')
    profiles, refs, predictions = [read(BASE / name) for name in ('profiles.json', 'references.json', 'predictions.json')]
    baseline = read(BASE / 'critical_fields.json')
    for key, value in [('profile_sha256', profiles), ('reference_sha256', refs), ('predictions_sha256', predictions)]:
        if fingerprint(value) != baseline['settings'][key]:
            raise ValueError(f'Frozen pilot input changed: {key}')
    for methods in data.values():
        for method in EXCLUDED_METHODS:
            methods.pop(method, None)
    for method in EXCLUDED_METHODS:
        predictions['methods'].pop(method, None)
    manifest_path = root / 'manifest.json'
    manifest = read(manifest_path)
    if manifest['failures'] or len(manifest['documents']) != 3:
        raise ValueError('Require a complete three-document OpenAI run')
    documents, configs, timing, provenance = {}, [], [], {}
    for doc, filename in SOURCES.items():
        records = [r for r in manifest['documents'] if r['source'] == filename]
        if len(records) != 1:
            raise ValueError(f'{doc}: require exactly one selected OpenAI run')
        responses, config = verify_openai(records[0], root, doc)
        configs.append(config)
        provenance[doc] = records[0]
        markdown = '\n\n'.join(r['markdown'] for r in responses)
        data[doc][MODEL] = {'pages': [markdown], 'input_format': 'markdown'}
        text = field_text(markdown, 'markdown')
        (out / (doc + '-decoded.txt')).write_text(text, encoding='utf-8')
        documents[doc] = {'fields': extract(doc, text)}
        for r in responses:
            raw = r['raw_response']
            usage = raw.get('usage') or {}
            timing.append({'document': doc, 'page': r['page_number'], 'seconds': r['seconds'],
                           'returned_model': raw['model'], 'response_id': raw['id'],
                           'input_tokens': usage.get('input_tokens', 0),
                           'output_tokens': usage.get('output_tokens', 0),
                           'reasoning_tokens': (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0)})
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError('OpenAI documents use different configurations')
    predictions['methods'][MODEL] = {'family': 'openai-vlm', 'documents': documents,
                                    'provenance': {'manifest': str(manifest_path), 'sha256': sha(manifest_path),
                                                   'config': configs[0], 'runs': provenance}}
    fields = evaluate_benchmark(profiles, refs, predictions, split='validation')
    fields['meaning'] = baseline['meaning']
    fields['extractor_version'] = EXTRACTOR_VERSION
    fields['pilot_limitations'] = baseline['pilot_limitations']
    for method in baseline['methods'].keys() - EXCLUDED_METHODS:
        if fields['methods'][method]['score'] != baseline['methods'][method]['score']:
            raise ValueError('Frozen baseline pilot score changed')
    audit = [{'method': method, 'document': doc, 'field': f['field'],
              'source_page': f['reference_location']['page'], 'reference': f['reference_value'],
              'prediction': f['predicted_value'], 'result': f['reason']}
             for method, scored in fields['methods'].items() for doc, result in scored['documents'].items()
             for f in result['field_results']]
    write_csv(out / 'field_audit.csv', audit)
    write_csv(out / 'page_timing.csv', timing)
    write_csv(out / 'critical_field_ranking.csv', fields['ranking'])
    print('Critical fields:', json.dumps(fields['ranking'], indent=2), flush=True)
    print('Computing whole-text comparison without ontology...', flush=True)
    result, pairs, volumes, normalized = evaluate(data, ontology_path=None, overlap_weight=1.0, weighting='document')
    result['comparison_unit'] = 'whole_document; page=1 is a matrix index, not a physical page'
    result['excluded_methods'] = sorted(EXCLUDED_METHODS)
    families = {m: f for m, f in baseline_manifest['families'].items() if m not in EXCLUDED_METHODS}
    families[MODEL] = 'openai-vlm'
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
               'source_sha256': SOURCE_SHA256, 'evaluation_script_sha256': sha(Path(__file__)),
               'runner_script_sha256': sha(ROOT / 'scripts/run_astra_ocr.py'),
               'extractor_version': EXTRACTOR_VERSION, 'families': families,
               'excluded_methods': sorted(EXCLUDED_METHODS)})
    make_report(out, result, fields, timing, manifest, configs[0], sensitivity)
    print(out / 'report.md', flush=True)


def make_report(out, result, fields, timing, manifest, config, sensitivity):
    lines = ['# Verified OpenAI VLM experiment: comparison without ontology', '',
             f'All 48 pages were transcribed through the official OpenAI Responses API. Every saved response '
             f'reports `{MODEL}` and completed status. The historical `openai-6` output is excluded from this comparison.', '',
             f'Settings: {config["dpi"]} DPI, image detail `{config["image_detail"]}`, reasoning `{config["reasoning"]}`, '
             f'maximum output {config["max_output_tokens"]} tokens/page, structured JSON, `store=false`. '
             'Pages were supplied as images without prior OCR, benchmark values or manual transcription corrections.', '',
             '## Whole-document token overlap', '',
             f'No ontology or sequence weighting. Average agreement with {len(result["models"]) - 1} peers, then average the three documents equally. '
             'These scores measure peer consensus, not correctness. Changing the peer set changes the scores. '
             'Whole-document comparison accommodates Insavlo’s missing page boundaries.', '',
             '| Rank | Saved output | Overall / 100 | CSA | Board | Trade | Family-balanced |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for row in result['ranking']:
        m = row['model']
        scores = [next(x['consensus_score'] for x in result['per_document_rankings'][d] if x['model'] == m)
                  for d in ('csa', 'board', 'trade')]
        lines.append(f'| {row["rank"]} | {m} | {row["consensus_score"]:.2f} | ' +
                     ' | '.join(f'{s:.2f}' for s in scores) + f' | {sensitivity["family_balanced_scores"][m]:.2f} |')
    lines += ['', '## Unchanged critical-field pilot', '',
              f'The same 27 selected fields and shared extractor version {EXTRACTOR_VERSION} are used for every method. '
              'Formatting tolerance preserves raw spans; reference values and comparison policies are unchanged. This is an exploratory OCR-plus-extraction '
              'pilot with Codex image-checked references, not independently adjudicated or held-out OCR accuracy. '
              'Missing spaces, labels or table structure can cause extraction failures even when values are transcribed.', '',
              '| Output | Score / 100 | Correct / 27 | Board / 13 | CSA / 6 | Trade / 8 |',
              '|---|---:|---:|---:|---:|---:|']
    for rank in fields['ranking']:
        m = rank['method']; r = fields['methods'][m]
        scores = [r['documents'][d]['counts']['critical_correct'] for d in ('board', 'csa', 'trade')]
        lines.append(f'| {m} | {rank["critical_field_score"]:.2f} | {r["metrics"]["critical_correct"]} | ' +
                     ' | '.join(map(str, scores)) + ' |')
    lines += ['', '## Runtime and provenance', '',
              f'- Run wall time: {manifest["wall_seconds"]:.2f} seconds with {manifest["workers"]} concurrent document workers. '
              'This includes rendering, requests/retries and saving; resumed runs may reuse checkpoints.',
              f'- Summed page request time: {sum(r["seconds"] for r in timing):.2f} seconds. Requests overlap, so this is not wall time.',
              f'- API usage: {sum(r["input_tokens"] for r in timing):,} input tokens; '
              f'{sum(r["output_tokens"] for r in timing):,} output tokens, including '
              f'{sum(r["reasoning_tokens"] for r in timing):,} reasoning tokens. Actual billing is provider-reported separately.',
              '- Raw responses, requested/returned model IDs, response IDs, page image hashes and source/settings hashes are retained. No API credentials are saved.',
              '- fable-5-1 is a historical Tesseract-based output label. The gpt-6-astra run is a verified OpenAI VLM family. Historical openai-6 artifacts are retained on disk but excluded from all scored outputs here.',
              '- Baseline limitations remain: Insavlo engine identity and source provenance are unverified; Fable source hashes are unverified; Mistral trade used an earlier TIFF; DeepSeek trade page 10 was truncated.', '',
              '## Reproduce', '', '```bash',
              'conda run --no-capture-output -n myenv3.13 python scripts/run_astra_ocr.py --workers 3 --output output/openai-vlm-gpt-6-astra',
              'conda run --no-capture-output -n myenv3.13 python scripts/evaluate_openai_vlm.py', '```', '',
              'Set OPENAI_API_KEY locally for inference. Evaluation uses saved responses only. See ranking.csv, '
              'field_audit.csv, page_timing.csv, and run_manifest.json in this directory.', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
