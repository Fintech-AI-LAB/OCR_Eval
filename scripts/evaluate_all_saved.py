"""One reproducible, offline comparison of all selected saved OCR workflows.

Never invokes inference. Historical experiments remain untouched. The primary
reference track requires physical page identity; unaligned text is diagnostic.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import statistics
import sys

from evaluate_six_critical_fields import (ROOT, SOURCES, SOURCE_SHA256, SAVED,
    annotations, extract, extract_pages, sha, PAGE_EXTRACTOR_VERSION)
from evaluate_paddleocr_v4 import PAGE_COUNTS, verify_run as verify_paddle
from evaluate_openai_vlm import verify_openai
from evaluate_claude_vlm import verify_claude
from evaluate_unlimited_ocr import verify_unlimited
from run_claude_ocr import validate_checkpoint, MODEL as CLAUDE

sys.path.insert(0, str(ROOT / 'standalone_ocr_eval'))
from ocr_eval import evaluate_benchmark
from ocr_eval.field_values import field_text
from ocr_eval.profiles import fingerprint
from ocr_eval.unified import compare_saved
from ocr_eval.corroboration import (corroborate_saved, corroboration_policy, passage_csv_rows,
                                   DEFAULT_MIN_FAMILIES, DEFAULT_MIN_FRACTION)

VERSION = '1.1'
# Explicit, versioned voters. Candidate variants and historical reviewed outputs
# never enter this panel merely because another saved workflow is added.
PANEL = ['mistral', 'mineru', 'deepseek', 'paddleocr-v4', 'gpt-6-astra',
         'claude-fable-5-1', 'unlimited-ocr-bf16']
RENAMES = {'openai-6': 'tesseract-reviewed-openai-label',
           'fable-5-1': 'tesseract-fable-label'}
FAMILIES = {'mistral': ['mistral'], 'mineru': ['mineru'], 'insavlo': ['unknown-insavlo'],
    'deepseek': ['deepseek'], 'paddleocr-v4': ['paddle'], 'gpt-6-astra': ['openai'],
    'claude-fable-5-1': ['claude'], 'unlimited-ocr-bf16': ['deepseek'],
    'unlimited-ocr-bf16-rotated': ['deepseek'],
    'tesseract-reviewed-openai-label': ['tesseract'], 'tesseract-fable-label': ['tesseract'],
    'claude-fable-5-1+tesseract': ['claude', 'tesseract']}


def evaluator_hashes():
    names = ['evaluate_all_saved', 'evaluate_six_critical_fields', 'evaluate_openai_vlm',
             'evaluate_paddleocr_v4', 'evaluate_claude_vlm', 'evaluate_unlimited_ocr',
             'run_astra_ocr', 'run_claude_ocr', 'run_unlimited_ocr', 'run_paddleocr_v4', 'ocr_common']
    paths = [ROOT/'scripts'/f'{name}.py' for name in names]
    paths += sorted((ROOT/'standalone_ocr_eval/ocr_eval').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def csv_write(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_saved():
    data = {doc: {} for doc in SOURCES}
    artifacts, notes = {}, []

    def track(path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Artifact changed: {path}')
        artifacts[str(path.relative_to(ROOT))] = actual
        return path

    def put(doc, method, pages, fmt='markdown', statuses=None, **extra):
        if method in data[doc]:
            raise ValueError(f'Duplicate selected run: {doc}/{method}')
        if pages is not None and len(pages) != PAGE_COUNTS[doc]:
            raise ValueError(f'Physical page count mismatch: {doc}/{method}')
        states = statuses if statuses is not None else ['complete'] * PAGE_COUNTS[doc]
        if len(states) != PAGE_COUNTS[doc]:
            raise ValueError('Page status count mismatch')
        data[doc][method] = {'pages': pages, 'input_format': fmt, 'statuses': states, **extra}

    for doc, filename in SOURCES.items():
        track(ROOT / 'data' / filename, SOURCE_SHA256[doc])
    prior = read(track(SAVED / 'with_ontology/results.json'))['provenance']
    track(SAVED / 'input.json', prior['filtered_input_sha256'])
    legacy_records = prior['artifacts']
    raw_tesseract = {}
    for record in legacy_records:
        doc, label = record['document'], record['engine_label']
        method = RENAMES.get(label, label)
        for path, expected in record['artifact_sha256'].items():
            track(path, expected)
        paths = [Path(p) for p in record['artifacts']]
        payloads = [read(p) for p in paths if p.name != 'source.json'] if label != 'insavlo' else []
        fmt, statuses = 'markdown', None
        if label == 'insavlo':
            put(doc, method, None, document_text=paths[0].read_text(encoding='utf-8-sig'),
                statuses=['unaligned'] * PAGE_COUNTS[doc], source_verified=False)
            continue
        if label == 'mistral':
            pages = []
            for response in payloads:
                for page in response['pages']:
                    text = page['markdown']
                    for table in page.get('tables') or []:
                        text = text.replace(f'[{table["id"]}]({table["id"]})', table['content'])
                    pages.append(text)
        elif label == 'mineru':
            pages = ['\n\n'.join(b.get('content') or '' for b in blocks) for blocks in payloads]
        elif label == 'openai-6':
            raw_tesseract[doc] = payloads[0]
            pages = [p['markdown'] for p in payloads[0]['pages']]
        elif label == 'fable-5-1':
            pages = ['\n'.join(line['text'] for line in p['lines']) for p in payloads[0]['full_page_text']]
            fmt = 'plain'
        elif label == 'deepseek':
            if [p['page'] for p in payloads] != list(range(1, PAGE_COUNTS[doc] + 1)):
                raise ValueError('DeepSeek page order mismatch')
            pages = [p['markdown'] for p in payloads]
            statuses = ['truncated' if p.get('truncated') else 'complete' for p in payloads]
        else:
            raise ValueError(f'Unsupported saved method {label}')
        put(doc, method, pages, fmt, statuses, source_verified=record.get('source_hash_verified') is True)
    notes += ['Insavlo has unverified engine/source identity and no reliable physical page boundaries.',
              'Mistral trade used an earlier TIFF; physical ordering is inherited from the saved run, source equivalence is unverified.',
              'Historical openai-6 and fable-5-1 are Tesseract workflows, with review described in their metadata; neither is a verified named VLM run.',
              'DeepSeek trade page 10 is truncated; its saved text remains in full-corpus scoring.']

    openai_pages = {}
    for method, folder_name, verifier, text_key, fmt in [
        ('gpt-6-astra', 'openai-vlm-gpt-6-astra', verify_openai, 'markdown', 'markdown'),
        ('paddleocr-v4', 'paddleocr-v4', verify_paddle, 'text', 'plain'),
        ('unlimited-ocr-bf16', 'unlimited-ocr-bf16', verify_unlimited, 'markdown', 'markdown'),
        ('unlimited-ocr-bf16-rotated', 'unlimited-ocr-bf16-rotated', verify_unlimited, 'markdown', 'markdown'),
    ]:
        root = ROOT / 'output' / folder_name
        manifest = read(track(root / 'manifest.json'))
        if manifest['failures'] or len(manifest['documents']) != len(SOURCES):
            raise ValueError(f'Unexpected incomplete selected run: {method}')
        for doc, filename in SOURCES.items():
            record, = [r for r in manifest['documents'] if r['source'] == filename]
            if verifier is verify_openai:
                responses, _ = verifier(record, root, doc)
                openai_pages[doc] = responses
            elif verifier is verify_paddle:
                responses, _ = verifier(record, ROOT / 'data' / filename, root, PAGE_COUNTS[doc])
            else:
                responses, _ = verifier(record, root, ROOT / 'data' / filename, PAGE_COUNTS[doc])
            folder = root / record['run']
            for p in [folder / 'source.json', *sorted(folder.glob('response-*.json'))]:
                track(p)
            put(doc, method, [r[text_key] for r in responses], fmt, source_verified=True)

    root = ROOT / 'output/claude-vlm-fable-5-1'
    manifest = read(track(root / 'manifest.json'))
    for doc, filename in SOURCES.items():
        records = [r for r in manifest['documents'] if r['source'] == filename]
        failures = [r for r in manifest['failures'] if r['source'] == filename]
        if len(records) + len(failures) != 1:
            raise ValueError('Require exactly one selected Claude run per document')
        record = (records or failures)[0]
        folder = (root / record['run']).resolve()
        if not folder.is_relative_to(root.resolve()):
            raise ValueError('Claude run path escapes output root')
        if records:
            responses, _ = verify_claude(record, root, doc)
            track(folder / 'source.json')
        pages, statuses = [], []
        failed = {p['page_number'] for p in record.get('failed_pages', [])}
        expected_files = {f'response-{i:04d}.json' for i in range(1, PAGE_COUNTS[doc]+1) if i not in failed}
        if {p.name for p in folder.glob('response-*.json')} != expected_files:
            raise ValueError('Claude page checkpoint coverage mismatch')
        for page in range(1, PAGE_COUNTS[doc]+1):
            if page in failed:
                failure = read(track(folder / f'failed-{page:04d}.json'))
                if failure['page_number'] != page:
                    raise ValueError('Failure page mismatch')
                pages.append('')
                statuses.append('failed')
            else:
                response = read(track(folder / f'response-{page:04d}.json'))
                validate_checkpoint(response, page, CLAUDE)
                # Partial runs lack source.json. Bind every successful page to
                # the verified OpenAI run's image of the same physical page.
                if response['image_sha256'] != openai_pages[doc][page-1]['image_sha256']:
                    raise ValueError('Claude/OpenAI physical image mismatch')
                pages.append(response['markdown'])
                statuses.append('complete')
        put(doc, CLAUDE, pages, statuses=statuses, source_verified=True)
    notes.append('Claude identity is gateway-reported. CSA pages 2 and 3 failed; they remain empty in full-corpus scoring. Partial-run checkpoints are bound to matching verified OpenAI page image hashes and locked in this manifest.')

    # Reconstruct the already-saved hybrid from exactly its documented inputs.
    hybrid_root = ROOT / 'output/claude_tesseract_hybrid_comparison'
    hybrid = read(track(hybrid_root / 'manifest.json'))
    track(hybrid['tesseract_artifact'], hybrid['tesseract_artifact_sha256'])
    hybrid_input = read(track(hybrid_root / 'csa_input.json'))
    for doc in SOURCES:
        original = data[doc][CLAUDE]
        pages = list(original['pages'])
        if doc == 'csa':
            for page in hybrid['fallback_pages']:
                if original['statuses'][page-1] != 'failed':
                    raise ValueError('Hybrid fallback no longer matches failed pages')
                # Decode formats independently before mixing raw/plain fallback
                # and Markdown. This prevents plain angle brackets becoming HTML.
            pages = [field_text(p, 'markdown') if i+1 not in hybrid['fallback_pages']
                     else raw_tesseract[doc]['pages'][i]['raw_text'] for i,p in enumerate(pages)]
            saved = hybrid_input['csa']['claude-fable-5-1+tesseract']['pages'][0]
            raw_joined = '\n\n'.join(original['pages'][i] if i+1 not in hybrid['fallback_pages']
                else raw_tesseract[doc]['pages'][i]['raw_text'] for i in range(PAGE_COUNTS[doc]))
            if raw_joined != saved:
                raise ValueError('Reconstructed hybrid differs from saved hybrid text')
        else:
            pages = [field_text(p, 'markdown') for p in pages]
        put(doc, 'claude-fable-5-1+tesseract', pages, 'plain', source_verified=True)
    notes.append('Claude/Tesseract hybrid uses only the previously saved raw Tesseract fallback on CSA pages 2 and 3. Unlimited rotated is a separate post-hoc preprocessing variant; neither adds a panel vote.')
    for doc, records in data.items():
        if set(records) != set(FAMILIES):
            raise ValueError(f'Method inventory mismatch for {doc}')
    return data, artifacts, notes


def coverage_rows(data):
    return [{'method': method, 'document': doc, 'physical_page': page,
             'status': status, 'source_verified': r['source_verified']}
            for doc, methods in data.items() for method, r in methods.items()
            for page, status in enumerate(r['statuses'], 1)]


def run_reference(data, profiles, refs, mask=None):
    selected_refs = deepcopy(refs)
    # Keep profiles fixed; fields outside the shared page subset are explicitly
    # not applicable for every workflow rather than dropped per model.
    if mask is not None:
        for doc, reference in selected_refs['documents'].items():
            for record in reference['records']:
                for field, annotation in record['fields'].items():
                    if annotation['page']-1 not in mask[doc]:
                        record['fields'][field] = {'status': 'not_applicable'}
    methods = [m for m in FAMILIES if all(data[d][m]['pages'] is not None for d in data)]
    predictions = {'schema_version': 1, 'methods': {}}
    for method in methods:
        documents = {}
        for doc in data:
            record = data[doc][method]
            pages = [p if mask is None or i in mask[doc] else '' for i,p in enumerate(record['pages'])]
            documents[doc] = {'fields': extract_pages(doc, pages, record['input_format'])}
            if record['source_verified']:
                documents[doc]['source_sha256'] = SOURCE_SHA256[doc]
        predictions['methods'][method] = {'family': '+'.join(FAMILIES[method]), 'documents': documents}
    result = evaluate_benchmark(profiles, selected_refs, predictions, split='validation')
    result.update(meaning='Exploratory page-bound OCR-plus-extraction pilot; not held-out model accuracy.',
                  page_extractor_version=PAGE_EXTRACTOR_VERSION,
                  unavailable_methods={'insavlo': 'Physical page boundaries unavailable; excluded from page-bound accuracy.'})
    for method, scored in result['methods'].items():
        operational = []
        for doc, values in scored['documents'].items():
            record = data[doc][method]
            indices = range(len(record['statuses'])) if mask is None else mask[doc]
            complete = all(record['statuses'][i] == 'complete' for i in indices)
            passed = values['metrics']['all_critical_fields_correct_document_rate']
            values['complete_and_pilot_pass'] = bool(complete and passed == 1)
            operational.append(values['complete_and_pilot_pass'])
        scored['complete_and_pilot_pass_rate'] = statistics.mean(operational)
        doc_scores = [v['metrics']['weighted_critical_field_accuracy'] for v in scored['documents'].values()
                      if v['metrics']['weighted_critical_field_accuracy'] is not None]
        # A sensitivity range, explicitly not a confidence interval with n=3.
        scored['leave_one_document_out_scores'] = [100 * statistics.mean(doc_scores[:i]+doc_scores[i+1:])
                                                   for i in range(len(doc_scores))] if len(doc_scores)>1 else []
    return result, predictions, selected_refs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/unified_corroboration_2families_v2')
    parser.add_argument('--corroboration-min-families', type=int, default=DEFAULT_MIN_FAMILIES)
    parser.add_argument('--corroboration-min-fraction', type=float, default=DEFAULT_MIN_FRACTION,
                        help='Optional proportional support floor; default 0 requires only two families.')
    parser.add_argument('--passage-tokens', type=int, default=40)
    args = parser.parse_args()
    corroboration_settings = dict(min_families=args.corroboration_min_families,
                                  min_fraction=args.corroboration_min_fraction,
                                  passage_tokens=args.passage_tokens)
    policy = corroboration_policy(**corroboration_settings)
    code_hashes = evaluator_hashes()
    print('Verifying selected saved outputs and preserving physical page identity...', flush=True)
    data, artifacts, notes = load_saved()
    profiles, refs = annotations()
    aligned = {d: {m:r for m,r in records.items() if r['pages'] is not None} for d,records in data.items()}
    families = {m:FAMILIES[m] for m in next(iter(aligned.values()))}
    common = {d: [i for i in range(PAGE_COUNTS[d])
                  if all(r['statuses'][i] == 'complete' for r in records.values())] for d,records in aligned.items()}
    settings = {'version': VERSION, 'panel': PANEL, 'families': FAMILIES,
        'corroboration_policy': policy,
        'source_sha256': SOURCE_SHA256, 'expected_pages': PAGE_COUNTS,
        'profile_sha256': fingerprint(profiles), 'reference_sha256': fingerprint(refs),
        'page_extractor_version': PAGE_EXTRACTOR_VERSION,
        'full_policy': 'All 48 pages; failed pages are empty, truncated text retained, page availability reported separately.',
        'common_page_numbers': {d:[i+1 for i in indices] for d,indices in common.items()},
        'primary_metric': 'Page-bound reference critical-field accuracy with strict duplicate rejection.',
        'consensus_metric': 'Fixed-panel family-balanced 0.5 token Dice + 0.5 ordered bigram Dice.',
        'unaligned_policy': 'Document-level diagnostics only; never infer physical pages from matching reference values.',
        'excluded_saved_artifacts': {'extractor_v1_orientation_enabled_archive': 'Superseded Paddle settings and old score snapshots; active manifest selects current Paddle run.',
            'fable-5-1-vlm': 'Runner exists, but no saved native inference run is available.'}}
    manifest = {'settings': settings, 'artifacts': artifacts,
                'code_sha256': code_hashes,
                'input_sha256': fingerprint(data), 'notes': notes}
    out = args.output.resolve()
    if (out / 'run_manifest.json').exists() and read(out / 'run_manifest.json') != manifest:
        raise ValueError('Inputs, panel, or evaluator changed. Use a new --output directory to preserve the previous comparison.')
    print('Scoring full-corpus and common-page reference tracks...', flush=True)
    full, predictions, _ = run_reference(data, profiles, refs)
    subset, subset_predictions, subset_refs = run_reference(data, profiles, refs, common)
    full_agreement = compare_saved(aligned, families, PANEL)
    common_agreement = compare_saved(aligned, families, PANEL, page_mask=common)
    document_agreement = compare_saved(data, FAMILIES, PANEL, whole_document=True)
    print('Aligning passages and counting fixed-panel family support...', flush=True)
    corroboration = corroborate_saved(aligned, families, PANEL, **corroboration_settings)
    # Preserve an unbound diagnostic for Insavlo, without presenting it as a
    # comparable reference accuracy result.
    unbound = {d: extract(d, field_text(data[d]['insavlo']['document_text'], 'markdown')) for d in data}
    coverage = coverage_rows(data)
    rows = []
    for method in FAMILIES:
        scored = full['methods'].get(method)
        common_scored = subset['methods'].get(method)
        metrics = scored['metrics'] if scored else {}
        states = [r['status'] for r in coverage if r['method']==method]
        rows.append({'method':method, 'family':'+'.join(FAMILIES[method]),
            'pilot_score':scored['score'] if scored else None,
            'correct_fields':metrics.get('critical_correct'), 'reference_fields':metrics.get('critical_fields'),
            'value_precision':metrics.get('value_precision'),
            'pilot_document_pass_rate':metrics.get('all_critical_fields_correct_document_rate'),
            'complete_and_pilot_pass_rate':scored['complete_and_pilot_pass_rate'] if scored else None,
            'completed_pages':states.count('complete') if 'unaligned' not in states else None,
            'failed_pages':states.count('failed'), 'truncated_pages':states.count('truncated'),
            'common_pilot_score':common_scored['score'] if common_scored else None,
            'page_agreement':full_agreement['scores'].get(method),
            'common_page_agreement':common_agreement['scores'].get(method),
            'document_agreement':document_agreement['scores'][method]})
    rows.sort(key=lambda r: (r['pilot_score'] is None, -(r['pilot_score'] or 0), r['method']))
    if evaluator_hashes() != code_hashes or any(sha(ROOT/p) != h for p,h in artifacts.items()):
        raise ValueError('Inputs or evaluator changed during scoring; retry with a stable snapshot.')
    out.mkdir(parents=True, exist_ok=True)
    outputs = {'run_manifest':manifest, 'input':data, 'profiles':profiles, 'references':refs,
        'predictions':predictions, 'critical_fields':full, 'common_references':subset_refs,
        'common_predictions':subset_predictions, 'common_critical_fields':subset,
        'page_agreement':full_agreement, 'common_page_agreement':common_agreement,
        'document_agreement':document_agreement, 'insavlo_unbound_fields':unbound,
        'passage_corroboration':corroboration}
    for name, value in outputs.items():
        write(out / (name+'.json'), value)
    csv_write(out/'comparison.csv', rows)
    csv_write(out/'coverage.csv', coverage)
    passage_rows = list(passage_csv_rows(corroboration))
    csv_write(out/'passage_corroboration.csv', passage_rows)
    # Keep the CSV header even when no passages require review.
    for filename, selected in (
            ('uncertain_passages.csv', [r for r in passage_rows if r['status'] == 'uncertain']),
            ('flagged_passages.csv', [r for r in passage_rows if r['flags']])):
        with (out/filename).open('w', newline='', encoding='utf-8') as stream:
            if passage_rows:
                writer = csv.DictWriter(stream, fieldnames=list(passage_rows[0]))
                writer.writeheader()
                writer.writerows(selected)
    audit = [{'method':m, 'document':d, 'field':f['field'], 'page':f['reference_location']['page'],
              'reference':f['reference_value'], 'prediction':f['predicted_value'], 'result':f['reason'],
              'value_and_relationships_correct':f['value_and_relationships_correct']}
             for m,scored in full['methods'].items() for d,v in scored['documents'].items() for f in v['field_results']]
    csv_write(out/'field_audit.csv', audit)
    report(out, rows, notes, common, full, corroboration)
    print(json.dumps(rows, indent=2), flush=True)
    print(out/'report.md')


def report(out, rows, notes, common, full, corroboration):
    def number(value):
        return 'n/a' if value is None else f'{value:.2f}'
    lines = ['# Unified saved-output evaluation', '',
        'All selected saved workflows use one frozen reference pilot, page routing, duplicate policy, and consensus panel. No inference was run.', '',
        '**This is an exploratory 27-field pilot on seven pages from three documents, not held-out or independently adjudicated OCR accuracy.** '
        'Adding models does not enlarge this sample. Existing references are unchanged; independent annotations and additional documents remain necessary.', '',
        '## Full corpus: 48 physical pages', '',
        'Pilot score averages critical-field accuracy within each document, then weights the three documents equally. '
        'Every field is extracted on its assigned physical page. Multiple predictions for a field receive zero correctness credit. '
        'Failed pages remain empty; truncated outputs remain as saved. Pilot pass covers selected fields only. '
        'Complete + pilot pass additionally requires every page in that document to have completed inference; completion does not prove accurate transcription.', '',
        '| Workflow | Pilot /100 | Fields | Pilot documents passed | Complete + pilot pass | Completed pages | Page agreement /100 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fields = 'n/a' if r['correct_fields'] is None else f'{r["correct_fields"]}/{r["reference_fields"]}'
        passed = 'n/a' if r['pilot_document_pass_rate'] is None else f'{round(r["pilot_document_pass_rate"]*3)}/3'
        complete = 'n/a' if r['complete_and_pilot_pass_rate'] is None else f'{round(r["complete_and_pilot_pass_rate"]*3)}/3'
        pages = 'unaligned' if r['completed_pages'] is None else f'{r["completed_pages"]}/48'
        lines.append(f'| {r["method"]} | {number(r["pilot_score"])} | {fields} | {passed} | {complete} | {pages} | {number(r["page_agreement"])} |')
    lines += ['', 'Scores are tied when numerically equal; display order within ties is alphabetical, not a quality ranking. '
        'Historical Tesseract workflows include review and are supplemental workflows, not independently verified VLM results. '
        'Insavlo is unranked on page-bound accuracy because its page identity is unavailable.', '',
        '## Common completed pages', '',
        f'The identical {sum(map(len,common.values()))}-page subset is applied to every page-aligned workflow. '
        'This excludes any page failed or truncated by any selected workflow; it is a conditional diagnostic, not a replacement for full-corpus results.', '',
        '| Workflow | Common pilot /100 | Common page agreement /100 |', '|---|---:|---:|']
    lines += [f'| {r["method"]} | {number(r["common_pilot_score"])} | {number(r["common_page_agreement"])} |' for r in rows]
    lines += ['', 'Excluded physical pages: ' + '; '.join(f'{d}: '+', '.join(str(i+1) for i in range(PAGE_COUNTS[d]) if i not in common[d])
        for d in common if len(common[d])!=PAGE_COUNTS[d]) + '.', '',
        '## Agreement policy and fixed panel', '',
        'Agreement averages token Dice and ordered-bigram Dice. It detects local word/value swaps but is still not exact value correctness or global reading-order accuracy. '
        'Normalization is independent per output. It requires no global sequence alignment. Each page is weighted equally within its document, then documents equally. '
        'Empty-empty pairs receive zero and pages never disappear from the full-corpus denominator.', '',
        'Frozen voters: ' + ', '.join(PANEL) + '. Peers sharing any engine family with the candidate are excluded; remaining peer scores are averaged within family and then equally across families. '
        'DeepSeek and both Unlimited variants share a conservative family. Hybrids exclude both constituent families. '
        'Adding a non-panel candidate leaves existing scores unchanged. Changing the panel is a new evaluation version; family balancing cannot eliminate shared-error bias.', '',
        'Whole-document diagnostics include all workflows, including unaligned Insavlo, with this same fixed panel:', '',
        '| Workflow | Document agreement /100 |', '|---|---:|']
    lines += [f'| {r["method"]} | {number(r["document_agreement"])} |' for r in rows]
    lines += ['', 'The page and document agreement columns use different units and must not be mixed. '
        'Leave-one-panel-family-out scores are retained in the agreement JSON files.', '',
        '## Passage corroboration and uncertain text', '',
        f'Each normalized source passage needs at least {corroboration["required_family_count"]} of '
        f'{len(corroboration["family_members"])} configured panel families. This configurable threshold is an uncalibrated review rule. '
        'Every configured member of a family must complete the page and support the same aligned token sequence for that family to vote. '
        'The source contributes only when it is a panel member; other_family_support_count excludes its family ancestry. '
        'Missing or failed families remain in the denominator. Non-panel variants never add votes.', '',
        'Meeting the family support threshold is sufficient for corroboration, even if other families disagree. '
        'Conflicting readings, ambiguous repeated-text alignment and incomplete source outputs remain separate flags and do not veto corroboration. '
        'Matching ignores case, formatting and ordinary punctuation; signed numeric tokens are preserved. '
        'Passages follow decoded source lines, split at the configured token limit. They can differ between workflows, so counts are not a model ranking. '
        'Alignment is textual within physical pages, not image-coordinate verification. Corroboration cannot prove correctness or detect text omitted by every model. '
        'Empty outputs remain uncertain; they are not verified blank pages.', '',
        '[All passage counts and flags](passage_corroboration.csv) · '
        '[Uncertain passages for review](uncertain_passages.csv) · '
        '[Flagged passages, including corroborated text](flagged_passages.csv) · '
        '[Per-family evidence and alternate readings](passage_corroboration.json)', '',
        '| Workflow | Corroborated passages | Uncertain passages | Flagged passages (either status) | Empty page outputs |',
        '|---|---:|---:|---:|---:|']
    for method in sorted({p['method'] for p in corroboration['pages']}):
        pages = [p for p in corroboration['pages'] if p['method'] == method]
        uncertain = sum(p['uncertain_passages'] for p in pages)
        lines.append(f'| {method} | {sum(p["passages"] for p in pages) - uncertain} | '
                     f'{uncertain} | {sum(p["flagged_passages"] for p in pages)} | {sum(p["empty_output"] for p in pages)} |')
    lines += ['',
        '## Sample and provenance limits', '', *['- '+n for n in notes], '',
        'No statistical significance or confidence interval is claimed from three documents. '
        'Leave-one-document-out pilot sensitivity scores are saved per workflow in critical_fields.json. '
        'Hashes prove artifact identity, not upstream model identity or reference correctness. '
        'The manifest records selected native artifacts, source checks, code hashes, panel and exact shared page subset. '
        'Changed inputs or code require a new output directory; existing experiments are preserved.', '',
        'See comparison.csv, coverage.csv and field_audit.csv for machine-readable results. '
        'The reference-bound track deliberately leaves Insavlo unavailable; insavlo_unbound_fields.json retains extraction diagnostics without page-bound accuracy credit.', '',
        '## Reproduce', '', '```bash',
        'conda run --no-capture-output -n myenv3.13 python scripts/evaluate_all_saved.py '
        f'--output "{out}" --corroboration-min-families {corroboration["policy"]["min_families"]} '
        f'--corroboration-min-fraction {corroboration["policy"]["min_fraction"]} '
        f'--passage-tokens {corroboration["policy"]["passage_tokens"]}', '```', '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
