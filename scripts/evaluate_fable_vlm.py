"""Evaluate the frozen claude-fable-5-1 VLM OCR run after inference is complete.

Adds `fable-5-1-vlm` as a distinct method beside the six frozen outputs. The
historical `fable-5-1` (Tesseract-based) baseline is kept unchanged. Reuses the
frozen 27-field pilot references and versioned shared extraction rules, and compares
whole documents WITHOUT the ontology using token overlap with equal document
weighting. Peer agreement is consensus, not ground-truth accuracy.

Refuses to run unless the manifest selects complete, hash-verified runs for all
48 pages and every page's provenance supports the claimed model.
"""
import argparse
import csv
import difflib
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

from ocr_common import ROOT, write_json
import run_fable_vlm_ocr as runner
from evaluate_six_critical_fields import (FAMILIES, SOURCES, SOURCE_SHA256,
                                         SAVED, OUT as PILOT, annotations, extract, sha,
                                         EXTRACTOR_VERSION)

sys.path.insert(0, str(ROOT / 'standalone_ocr_eval'))
from ocr_eval import evaluate, evaluate_benchmark, consensus_family_sensitivity
from ocr_eval.profiles import fingerprint
from ocr_eval.field_values import field_text

METHOD = runner.METHOD
FAMILY = 'anthropic-vlm'
PAGE_COUNTS = {'csa': 14, 'board': 4, 'trade': 30}
LIST_PRICE_PER_MTOK = {'input': 10.0, 'output': 50.0}  # documented Claude Fable 5.1 pricing


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------- verification

def verify_preflight(root, reference):
    path = (root / reference['path']).resolve()
    if not path.is_relative_to(root) or sha(path) != reference['sha256']:
        raise ValueError(f'Preflight evidence missing or changed: {reference["path"]}')
    report = read(path)
    if (report.get('status') != 'passed' or report.get('provider') != runner.PROVIDER
            or report.get('api_base') != runner.API_BASE or report.get('requested_model') != runner.MODEL):
        raise ValueError(f'Preflight did not verify {runner.MODEL} on {runner.API_BASE}: {reference["path"]}')
    retrieved = report['model_verification']['retrieved']
    if not retrieved.get('id'):
        raise ValueError('Preflight has no retrieved model ID')
    return report


def verify_run(record, source, root, expected_pages, accepted_models):
    """Reject partial, reordered, stale, modified or unprovenanced OCR before scoring."""
    root = Path(root).resolve()
    if record['source_sha256'] != sha(source):
        raise ValueError(f'Source hash mismatch: {source}')
    folder = (root / record['run']).resolve()
    if not folder.is_relative_to(root):
        raise ValueError('Run path escapes output root')
    if sha(folder / 'source.json') != record['source_metadata_sha256']:
        raise ValueError('source.json changed after selection')
    metadata = read(folder / 'source.json')
    if metadata['source'] != Path(source).name or metadata['method'] != METHOD:
        raise ValueError('Source/method metadata mismatch')
    config = metadata['config']
    if config != runner.inference_config():
        raise ValueError('Run settings differ from the specified model/prompt/settings')
    if folder.name != runner.cache_key(source, config):
        raise ValueError('Source/settings cache identity mismatch')
    for rel, expected in record['artifact_sha256'].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(folder) or sha(path) != expected:
            raise ValueError(f'Artifact missing or changed: {rel}')
    present = {runner.relative(p, root) for p in folder.rglob('*')
               if p.is_file() and not p.name.endswith('.tmp') and p.name != '.DS_Store'}
    if present - set(record['artifact_sha256']):
        raise ValueError(f'Unselected files inside the run folder: {sorted(present - set(record["artifact_sha256"]))[:5]}')
    paths = sorted((folder / 'pages').glob('page-*.json'))
    if (len(paths) != expected_pages or record['pages'] != expected_pages
            or metadata['page_count'] != expected_pages):
        raise ValueError('Incomplete or extra page checkpoints')
    records = []
    for page, path in enumerate(paths, 1):
        if path.name != f'page-{page:04d}.json':
            raise ValueError('Page order mismatch')
        r = read(path)
        runner.validate_page_record(r, folder, page, config, accepted_models)
        if r['source_sha256'] != record['source_sha256'] or r['page_count'] != expected_pages:
            raise ValueError('Page belongs to a different source or page count')
        records.append(r)
    return records, metadata


# ------------------------------------------------------------ diagnostics

def collapse(value):
    return ' '.join(str(value).split())


def alnum(value):
    return re.sub(r'[\W_]+', '', str(value).casefold())


def format_key(value):
    """Drop spacing, grouping commas and case, but keep signs and decimal points."""
    return re.sub(r'[^\w.\-]+', '', str(value).casefold())


def canon(value):
    """Compare live results with JSON-loaded frozen ones on equal footing (tuples, keys)."""
    return json.loads(json.dumps(value))


def diagnose(reference, predicted, page_text, document_text):
    """Heuristic split of a failed field into OCR-side vs parser/binding-side causes."""
    if reference is None or not collapse(reference):
        return 'no_reference_value', ''
    weak = 'weak: short value' if len(alnum(reference)) <= 3 else ''
    if collapse(reference) in collapse(page_text):
        return 'parser_or_field_binding_failure', weak
    if format_key(reference) and format_key(reference) in format_key(page_text):
        return 'format_or_punctuation_difference', weak
    if predicted and difflib.SequenceMatcher(None, alnum(predicted), alnum(reference)).ratio() >= 0.75:
        return 'ocr_transcription_error_near_miss', weak
    if alnum(reference) and alnum(reference) in alnum(document_text):
        return 'value_only_on_another_page', weak
    return 'ocr_omission_or_misread', weak


DIAGNOSIS_MEANING = {
    'parser_or_field_binding_failure': 'Exact reference value is in the VLM page text; the shared extractor did not bind it.',
    'format_or_punctuation_difference': 'Reference characters are present but spacing/punctuation/case differ (OCR formatting or parser normalisation; inspect).',
    'ocr_transcription_error_near_miss': 'Bound value is a close variant of the reference (likely misread characters).',
    'value_only_on_another_page': 'Value absent from the reference page but present elsewhere in the document.',
    'ocr_omission_or_misread': 'Value not found on the page, even ignoring punctuation (omitted or misread).',
}


# ------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ocr-output', type=Path, default=runner.OUTPUT)
    parser.add_argument('--output', type=Path, default=runner.OUTPUT / 'evaluation')
    args = parser.parse_args()
    root, output = args.ocr_output.resolve(), args.output.resolve()
    if {SOURCES[d]: PAGE_COUNTS[d] for d in SOURCES} != runner.EXPECTED_PAGES:
        raise ValueError('Pilot sources and runner page expectations disagree')

    # 1. Verify the frozen VLM OCR completely before reading any reference.
    manifest_path = root / 'manifest.json'
    manifest = read(manifest_path)
    if (not manifest.get('complete') or manifest.get('incomplete') or len(manifest['documents']) != len(SOURCES)
            or manifest['requested_model'] != runner.MODEL or manifest['provider'] != runner.PROVIDER
            or manifest['api_base'] != runner.API_BASE):
        raise ValueError('Require a complete three-document claude-fable-5-1 manifest from the first-party API')
    vlm_pages, vlm_records, preflights, metadata_records = {}, {}, {}, {}
    for doc, filename in SOURCES.items():
        selected = [r for r in manifest['documents'] if r['source'] == filename]
        if len(selected) != 1:
            raise ValueError(f'{doc}: require exactly one selected run')
        record = selected[0]
        if record['source_sha256'] != SOURCE_SHA256[doc]:
            raise ValueError('VLM source differs from the frozen pilot source')
        records, metadata = verify_run(record, ROOT / 'data' / filename, root, PAGE_COUNTS[doc],
                                       manifest['accepted_returned_models'])
        for r in records:
            key = r['preflight']['path']
            if key not in preflights:
                preflights[key] = verify_preflight(root, r['preflight'])
            retrieved = preflights[key]['model_verification']['retrieved']['id']
            if r['response']['returned_model'] not in {runner.MODEL, retrieved}:
                raise ValueError(f'{doc} page {r["page_number"]}: returned model not verified by its preflight')
        vlm_records[doc], metadata_records[doc] = records, metadata
        vlm_pages[doc] = [r['transcription'] for r in records]
    ids = [r['response']['id'] for rs in vlm_records.values() for r in rs]
    if len(ids) != sum(PAGE_COUNTS.values()) or len(set(ids)) != len(ids) or not all(ids):
        raise ValueError('Expected 48 pages with distinct provider response IDs')
    print(f'Verified {len(ids)} pages from {runner.MODEL} on {runner.API_BASE}.', flush=True)

    # 2. Frozen six-method inputs and pilot (unchanged), as in the PaddleOCR experiment.
    output.mkdir(parents=True, exist_ok=True)
    data = read(SAVED / 'input.json')
    prior = read(SAVED / 'with_ontology/results.json')
    provenance = prior['provenance']
    if sha(SAVED / 'input.json') != provenance['filtered_input_sha256']:
        raise ValueError('Frozen six-method input changed')
    verified = {}
    for record in provenance['artifacts']:
        for path, expected in record.get('artifact_sha256', {}).items():
            artifact = Path(path) if Path(path).is_absolute() else ROOT / path
            if sha(artifact) != expected:
                raise ValueError(f'Baseline OCR artifact changed: {path}')
            verified[path] = expected
    if METHOD in FAMILIES or any(METHOD in records for records in data.values()):
        raise ValueError('Method name collides with a frozen method')
    for doc, records in data.items():
        if any(len(r['pages']) != 1 for r in records.values()):
            raise ValueError('Frozen inputs are expected to hold one whole-document text per method')
    profiles, refs = annotations()  # includes fixed source PDF hash checks
    baseline = read(PILOT / 'results.json')
    if baseline.get('extractor_version') != EXTRACTOR_VERSION:
        raise ValueError('Reevaluate all six baselines with the current shared extractor first')
    if (fingerprint(profiles) != baseline['settings']['profile_sha256'] or
            fingerprint(refs) != baseline['settings']['reference_sha256']):
        raise ValueError('Pilot profiles or references changed')
    predictions = read(PILOT / 'predictions.json')
    if fingerprint(predictions) != baseline['settings']['predictions_sha256']:
        raise ValueError('Frozen pilot predictions changed')
    for method in FAMILIES:
        for doc in SOURCES:
            text = field_text('\n\n'.join(data[doc][method]['pages']), 'markdown')
            if canon(extract(doc, text)) != predictions['methods'][method]['documents'][doc]['fields']:
                raise ValueError('Pilot extraction rules no longer reproduce baseline predictions')

    # 3. Apply the same shared extractor to the new VLM output (Markdown with HTML tables).
    vlm_docs, decoded_pages, decoded_docs, timing, uncertainty_rows = {}, {}, {}, [], []
    decoded_dir = output / 'decoded' / METHOD
    decoded_dir.mkdir(parents=True, exist_ok=True)
    for doc in SOURCES:
        raw_text = '\n\n'.join(vlm_pages[doc])
        data[doc][METHOD] = {'pages': [raw_text], 'input_format': 'markdown'}
        text = field_text(raw_text, 'markdown')
        (decoded_dir / (doc + '.txt')).write_text(text, encoding='utf-8')
        decoded_docs[doc] = text
        decoded_pages[doc] = [field_text(p, 'markdown') for p in vlm_pages[doc]]
        fields = extract(doc, text)
        for item in fields:
            a, b = item['decoded_span']
            if item['value'] != text[a:b]:
                raise ValueError('Extractor changed a captured OCR value')
        vlm_docs[doc] = {'fields': fields}
        for r in vlm_records[doc]:
            usage = r['response'].get('usage') or {}
            timing.append({'document': doc, 'page': r['page_number'], 'seconds': r['timing']['seconds'],
                           'first_event_seconds': r['timing'].get('first_event_seconds'),
                           'input_tokens': usage.get('input_tokens', 0), 'output_tokens': usage.get('output_tokens', 0),
                           'image_width': r['render']['width'], 'image_height': r['render']['height'],
                           'transcription_characters': len(r['transcription']),
                           'uncertainties': len(r['uncertainties']), 'workers': r.get('workers'),
                           'response_id': r['response']['id'], 'request_id': r['response'].get('request_id')})
            uncertainty_rows += [{'document': doc, 'page': r['page_number'], 'uncertainty': u} for u in r['uncertainties']]
    predictions['methods'][METHOD] = {
        'family': FAMILY, 'documents': vlm_docs,
        'provenance': {'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
                       'model': runner.MODEL, 'provider': runner.PROVIDER, 'api_base': runner.API_BASE,
                       'extractor_version': EXTRACTOR_VERSION,
                       'extraction': 'Shared formatting-tolerant section/label rules; markdown/HTML decoding; raw spans; no value repair.'},
    }
    result = evaluate_benchmark(profiles, refs, predictions, split='validation')
    for method in FAMILIES:
        if (canon(result['methods'][method]['score']) != baseline['methods'][method]['score'] or
                canon(result['methods'][method]['metrics']) != baseline['methods'][method]['metrics']):
            raise ValueError('Baseline pilot scores changed; investigate before comparing')
    result['meaning'] = baseline['meaning']
    result['extractor_version'] = EXTRACTOR_VERSION
    result['pilot_limitations'] = baseline['pilot_limitations']
    field_totals = {doc: len(r['records'][0]['fields']) for doc, r in refs['documents'].items()}
    rows, audit = [], []
    for rank in result['ranking']:
        method = rank['method']
        scored = result['methods'][method]
        rows.append({**rank, 'correct': scored['metrics']['critical_correct'],
                     'total': scored['metrics']['critical_fields'],
                     **{d + '_correct': r['counts']['critical_correct'] for d, r in scored['documents'].items()},
                     'missing': scored['metrics']['missing_fields'], 'wrong': scored['metrics']['wrong_values']})
        for doc, r in scored['documents'].items():
            for f in r['field_results']:
                diagnosis = strength = ''
                if method == METHOD and f['reason'] != 'correct':
                    page = f['reference_location']['page']
                    diagnosis, strength = diagnose(f['reference_value'], f['predicted_value'],
                                                   decoded_pages[doc][page - 1], decoded_docs[doc])
                audit.append({'method': method, 'document': doc, 'field': f['field'],
                              'source_page': f['reference_location']['page'], 'reference': f['reference_value'],
                              'prediction': f['predicted_value'], 'result': f['reason'],
                              'vlm_failure_diagnosis': diagnosis, 'diagnosis_evidence': strength})
    for name, value in [('profiles.json', profiles), ('references.json', refs), ('predictions.json', predictions),
                        ('critical_fields.json', result), ('input.json', data)]:
        write_json(output / name, value)
    write_csv(output / 'critical_field_ranking.csv', rows)
    write_csv(output / 'field_audit.csv', audit)
    write_csv(output / 'page_timing.csv', timing)
    write_csv(output / 'uncertainties.csv', uncertainty_rows)

    # 4. Whole-document consensus WITHOUT ontology (token overlap, equal document weights).
    families = {**FAMILIES, METHOD: FAMILY}
    consensus, pairs, volumes, normalized = evaluate(data, ontology_path=None, weighting='document', overlap_weight=1.0)
    consensus['comparison_unit'] = 'whole_document; page=1 is a matrix index, not a physical page'
    consensus['meaning'] = 'Peer consensus (agreement with other OCR outputs), not ground-truth accuracy'
    sensitivity = consensus_family_sensitivity(pairs, families)
    six = {d: {m: r for m, r in records.items() if m != METHOD} for d, records in data.items()}
    six_consensus = evaluate(six, ontology_path=None, weighting='document', overlap_weight=1.0)[0]
    saved_six = SAVED / 'without_ontology' / 'results.json'
    six_reproduced = None
    if saved_six.exists():
        six_reproduced = read(saved_six).get('ranking') == canon(six_consensus['ranking'])
    folder = output / 'without_ontology'
    folder.mkdir(exist_ok=True)
    for name, value in [('results.json', consensus), ('pair_differences.json', pairs),
                        ('text_volume.json', volumes), ('normalized.json', normalized),
                        ('family_sensitivity.json', sensitivity), ('six_method_baseline_results.json', six_consensus)]:
        write_json(folder / name, value)
    write_csv(folder / 'ranking.csv', consensus['ranking'])
    write_csv(folder / 'per_document_ranking.csv',
              [{'document': d, **r} for d, ranking in consensus['per_document_rankings'].items() for r in ranking])

    run_configs = [m['config'] for m in metadata_records.values()]
    write_json(output / 'run_manifest.json', {
        'ocr_manifest': str(manifest_path), 'ocr_manifest_sha256': sha(manifest_path),
        'preflights': {k: {'sha256': sha(root / k), 'retrieved_model': v['model_verification']['retrieved']}
                       for k, v in preflights.items()},
        'baseline_input_sha256': sha(SAVED / 'input.json'), 'input_sha256': sha(output / 'input.json'),
        'pilot_predictions_sha256': sha(PILOT / 'predictions.json'),
        'evaluation_script_sha256': sha(Path(__file__)), 'runner_script_sha256': sha(Path(runner.__file__)),
        'extractor_script_sha256': sha(ROOT / 'scripts/evaluate_six_critical_fields.py'),
        'extractor_version': EXTRACTOR_VERSION,
        'source_sha256': SOURCE_SHA256, 'verified_baseline_artifacts': verified,
        'baseline_provenance': provenance, 'families': families,
        'six_method_consensus_reproduces_saved': six_reproduced,
    })
    make_report(output, rows, audit, timing, consensus, six_consensus, sensitivity, six_reproduced,
                preflights, vlm_records, run_configs[0], field_totals, uncertainty_rows)
    print(output / 'report.md', flush=True)


# ----------------------------------------------------------------- report

def cell(x):
    return str(x).replace('|', '\\|').replace('\n', ' ') if x not in (None, '') else '—'


def make_report(output, rows, audit, timing, consensus, six_consensus, sensitivity, six_reproduced,
                preflights, vlm_records, config, field_totals, uncertainty_rows):
    vlm = next(r for r in rows if r['method'] == METHOD)
    all_records = [r for rs in vlm_records.values() for r in rs]
    retrieved = {json.dumps(p['model_verification']['retrieved'], sort_keys=True) for p in preflights.values()}
    returned = sorted({r['response']['returned_model'] for r in all_records})
    sdk_versions = sorted({str(r['sdk'].get('version')) for r in all_records})
    workers = sorted({r.get('workers') for r in all_records}, key=str)
    total_fields = sum(field_totals.values())
    lines = [f'# {runner.MODEL} vision-language OCR experiment', '',
             f'All 48 physical pages were transcribed by `{runner.MODEL}` through the Anthropic Messages API. '
             f'Under shared extractor version {EXTRACTOR_VERSION} it gets {vlm["correct"]}/{total_fields} critical fields '
             f'(equal-document score {vlm["critical_field_score"]:.2f}/100).', '',
             '## Model provenance', '',
             f'- Provider: Anthropic, first-party endpoint `{runner.API_BASE}` (pinned; `ANTHROPIC_BASE_URL` ignored).',
             f'- Requested model ID: `{runner.MODEL}`. Returned model ID on every page: {", ".join(f"`{m}`" for m in returned)}.',
             '- Preflight `GET /v1/models/' + runner.MODEL + '` returned: ' + '; '.join(f'`{r}`' for r in sorted(retrieved)) + '.',
             f'- Anthropic Python SDK: {", ".join(sdk_versions)}. {len(all_records)} distinct response IDs; request IDs kept per page.',
             '- Every page passed re-validation against its preserved raw response (stop_reason `end_turn`, schema-valid JSON, accepted model).', '',
             '## Configuration', '',
             f'- Pages rendered at {config["dpi"]} DPI with the existing pypdfium2 renderer and sent as lossless PNG. '
             'No Tesseract, PaddleOCR, Apple Vision, PDF text layer or other OCR engine was used.',
             '- High-detail images: Anthropic has no `detail` parameter. Its documented high-resolution tier '
             '(up to 2576 px long edge / 4784 visual tokens) was enforced with `transformations.oversized_image = "error"`, '
             'so any image the model would have downscaled is rejected. The preflight token counter accepted every page size.',
             f'- Adaptive thinking (always on for this model), effort `{config["effort"]}`, max_tokens {config["max_tokens"]}, '
             'structured output `{transcription, uncertainties}`. Temperature/top_p/top_k cannot be set on this model, so '
             'output is not deterministic and a rerun may differ.',
             f'- Concurrency (documents in parallel): {", ".join(map(str, workers))}.',
             '- Prompt (verbatim, identical for every page):', '', '> ' + config['prompt'], '',
             '## Frozen critical-field pilot', '',
             f'The same {total_fields} references, profiles and shared extractor version {EXTRACTOR_VERSION} are used for all seven outputs. '
             'The six existing predictions are hash-verified and reproduced before adding the VLM. '
             'Fields are averaged within each document, then the three documents are weighted equally. '
             '`fable-5-1` is the historical Tesseract-based artifact; `fable-5-1-vlm` is this run.', '',
             f'| Method | Score / 100 | Correct / {total_fields} | Board / {field_totals["board"]} | CSA / {field_totals["csa"]} '
             f'| Trade / {field_totals["trade"]} | Unextracted | Wrong value/binding |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["method"]} | {r["critical_field_score"]:.2f} | {r["correct"]} | {r["board_correct"]} | '
                     f'{r["csa_correct"]} | {r["trade_correct"]} | {r["missing"]} | {r["wrong"]} |')
    failures = [r for r in audit if r['method'] == METHOD and r['result'] != 'correct']
    lines += ['', f'### {METHOD} field results', '',
              '| Document | Page | Field | Reference | Extracted | Result | Diagnosis |', '|---|---:|---|---|---|---|---|']
    for r in audit:
        if r['method'] == METHOD:
            diag = r['vlm_failure_diagnosis'] + (f' ({r["diagnosis_evidence"]})' if r['diagnosis_evidence'] else '')
            lines.append('| ' + ' | '.join(cell(x) for x in [r['document'], r['source_page'], r['field'], r['reference'],
                                                              r['prediction'], r['result'], diag]) + ' |')
    counts = {}
    for r in failures:
        counts[r['vlm_failure_diagnosis']] = counts.get(r['vlm_failure_diagnosis'], 0) + 1
    lines += ['', '**OCR errors vs parser/binding failures** (heuristic, checked only on the reference page of the VLM text after the same Markdown/HTML decoding):', '']
    if counts:
        lines += [f'- {k}: {v} — {DIAGNOSIS_MEANING.get(k, "")}' for k, v in sorted(counts.items())]
    else:
        lines.append('- No failed fields.')
    lines += ['', 'These labels award no credit and change no score. Short values (≤3 characters) are weak evidence. '
              'Confirm any label by viewing the saved page image under `output/fable-5-1-vlm/`.', '',
              '## Whole-document consensus WITHOUT ontology', '',
              '**This is peer consensus, not accuracy.** Each method is scored by mean token overlap with the other '
              'methods on whole documents (no ontology, overlap weight 1.0), with the three documents weighted equally. '
              'A method can agree with its peers and still be wrong. Adding a seventh method changes every peer average. '
              'openai-6 and fable-5-1 are both Tesseract-based artifacts, so the family-balanced column drops '
              'same-family peers.', '',
              '| Rank | Method | Token-overlap consensus | Family-balanced | Six-method baseline |',
              '|---:|---|---:|---:|---:|']
    six = {r['model']: r['consensus_score'] for r in six_consensus['ranking']}
    for r in consensus['ranking']:
        m = r['model']
        base = f'{six[m]:.2f}' if m in six else '—'
        lines.append(f'| {r["rank"]} | {m} | {r["consensus_score"]:.2f} | '
                     f'{sensitivity["family_balanced_scores"][m]:.2f} | {base} |')
    if six_reproduced is not None:
        lines += ['', f'Six-method baseline recomputed here {"matches" if six_reproduced else "DOES NOT match"} '
                  'the saved `without_ontology` ranking.']
    lines += ['', '### Per-document consensus (no ontology)', '']
    for doc, ranking in consensus['per_document_rankings'].items():
        lines.append(f'- **{doc}**: ' + ', '.join(f'{r["rank"]}. {r["model"]} {r["consensus_score"]:.2f}' for r in ranking))
    seconds = sum(r['seconds'] for r in timing)
    tokens_in = sum(r['input_tokens'] for r in timing)
    tokens_out = sum(r['output_tokens'] for r in timing)
    cost = tokens_in / 1e6 * LIST_PRICE_PER_MTOK['input'] + tokens_out / 1e6 * LIST_PRICE_PER_MTOK['output']
    lines += ['', '## Inference timing', '',
              f'Total page inference {seconds:.1f} s ({seconds / 60:.1f} min), mean {seconds / len(timing):.1f} s/page, '
              f'median {statistics.median(r["seconds"] for r in timing):.1f} s. Wall time per streamed request '
              '(network + queueing + thinking + generation); excludes rendering, preflight and evaluation. '
              f'Tokens: {tokens_in:,} input, {tokens_out:,} output (output includes hidden thinking); '
              f'about ${cost:.2f} at list price. Remote API timings are not comparable to the local engines.', '',
              '| Document | Pages | Seconds | Mean s/page | Max s/page | Input tokens | Output tokens | Uncertainty notes |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for doc, count in PAGE_COUNTS.items():
        rs = [r for r in timing if r['document'] == doc]
        duration = sum(r['seconds'] for r in rs)
        lines.append(f'| {doc} | {count} | {duration:.1f} | {duration / count:.1f} | {max(r["seconds"] for r in rs):.1f} | '
                     f'{sum(r["input_tokens"] for r in rs):,} | {sum(r["output_tokens"] for r in rs):,} | '
                     f'{sum(1 for u in uncertainty_rows if u["document"] == doc)} |')
    lines += ['', '## Limits', '',
              '- Exploratory pilot: 27 fields on seven source pages, not held-out accuracy or character error rate. '
              'References were image-checked by Codex, not independently human-adjudicated.',
              '- The fixed extraction rules were written for earlier outputs and may bind HTML-table output differently. '
              'Rules were not tuned to this run.',
              '- One sample per page with provider-default sampling. Model-reported uncertainties are in `uncertainties.csv`, '
              'kept separate from the transcription. OCR errors are left uncorrected.',
              '- Baseline provenance caveats (Fable/Insavlo hashes, Mistral TIFF, DeepSeek truncation) are unchanged; see `run_manifest.json`.', '',
              '## Reproduce', '', '```bash',
              'conda run --no-capture-output -n myenv3.13 python -m pip install -r requirements-anthropic.txt',
              'conda run --no-capture-output -n myenv3.13 python -m unittest tests.test_fable_vlm_ocr',
              'conda run --no-capture-output -n myenv3.13 python scripts/run_fable_vlm_ocr.py --preflight-only',
              'conda run --no-capture-output -n myenv3.13 python scripts/run_fable_vlm_ocr.py',
              'conda run --no-capture-output -n myenv3.13 python scripts/evaluate_fable_vlm.py', '```', '',
              'Raw runs: `output/fable-5-1-vlm/<pdf>/<cache-key>/` (`raw/` provider responses and stream events, '
              '`requests/`, `attempts/`, `images/`, `pages/` checkpoints and Markdown, `document.md`). '
              'Selection: `output/fable-5-1-vlm/manifest.json`. Model check: `output/fable-5-1-vlm/preflight/`.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
