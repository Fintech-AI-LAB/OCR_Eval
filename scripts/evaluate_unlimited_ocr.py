"""Compare complete Unlimited-OCR inference against saved full-corpus workflows."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

from ocr_common import ROOT, pages, write_json
from run_unlimited_ocr import METHOD, MODEL, REVISION, validate_response, sha, load_orientation, orient_page
from evaluate_paddleocr_v4 import PAGE_COUNTS, read, write_csv, verify_run
from evaluate_openai_vlm import verify_openai
from evaluate_six_critical_fields import SOURCES, SOURCE_SHA256, FAMILIES, extract, EXTRACTOR_VERSION

sys.path.insert(0, str(ROOT / 'standalone_ocr_eval'))
from ocr_eval import evaluate, evaluate_benchmark, consensus_family_sensitivity
from ocr_eval.field_values import field_text
from ocr_eval.profiles import fingerprint

BASE = ROOT / 'output/openai_vlm_experiment'
EXCLUDED = {'openai-6', 'fable-5-1'}


def command_path(path):
    path = path.resolve()
    return shlex.quote(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))


def verify_unlimited(record, root, source, expected):
    if sha(source) != record['source_sha256']:
        raise ValueError('Source hash mismatch')
    folder = (root / record['run']).resolve()
    if not folder.is_relative_to(root.resolve()):
        raise ValueError('Run path escapes output root')
    for name, expected_hash in record['artifact_sha256'].items():
        if Path(name).name != name or sha(folder / name) != expected_hash:
            raise ValueError('Artifact hash mismatch')
    metadata = read(folder / 'source.json')
    config = metadata['config']
    if (config['model'] != MODEL or config['revision'] != REVISION or
            config['backend'] != METHOD or metadata['status'] != 'complete' or
            metadata['source'] != record['source'] or metadata['source_sha256'] != sha(source)):
        raise ValueError('Model/source/status mismatch')
    digest = hashlib.sha256(source.read_bytes())
    digest.update(json.dumps(config, sort_keys=True).encode())
    if folder.name != digest.hexdigest()[:16]:
        raise ValueError('Source/settings directory hash mismatch')
    checkpoints = sorted(folder.glob('response-*.json'))
    required = {'source.json', 'document.md', *(f'response-{i:04d}.json' for i in range(1, expected + 1))}
    if (len(checkpoints) != expected or record['pages'] != expected or metadata['responses'] != expected or
            set(record['artifact_sha256']) != required):
        raise ValueError('Incomplete or extra page checkpoints')
    responses = []
    orientation = config.get('orientation')
    if orientation is not None:
        source_paths = {name: ROOT / 'data' / name for name in orientation['documents']}
        if load_orientation(Path(orientation['manifest']), source_paths, config['dpi']) != orientation:
            raise ValueError('Orientation provenance mismatch')
    for i, image in enumerate(pages(source, config['dpi']), 1):
        checkpoint = folder / f'response-{i:04d}.json'
        response = read(checkpoint)
        original_hash = hashlib.sha256(image.tobytes()).hexdigest()
        image, angle = orient_page(image, record['source'], i, orientation)
        if orientation is not None and (response.get('rotation_ccw_degrees') != angle or
                response.get('original_image_sha256') != original_hash or response['image_size'] != list(image.size)):
            raise ValueError('Rotation metadata mismatch')
        validate_response(response, i, hashlib.sha256(image.tobytes()).hexdigest())
        responses.append(response)
    if len(responses) != expected:
        raise ValueError('PDF page count mismatch')
    return responses, config


def verified_baselines(*, require_current_extractor=True):
    manifest = read(BASE / 'run_manifest.json')
    prior_path = Path(manifest['baseline_manifest'])
    if sha(prior_path) != manifest['baseline_manifest_sha256']:
        raise ValueError('Baseline manifest changed')
    prior = read(prior_path)
    if sha(prior_path.parent / 'input.json') != prior['input_sha256']:
        raise ValueError('Baseline input changed')
    for path, expected in prior['verified_baseline_artifacts'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Baseline artifact changed')
    verified_text = read(prior_path.parent / 'input.json')
    for m, verify in [(prior, 'paddle'), (manifest, 'openai')]:
        path = Path(m['ocr_manifest'])
        if sha(path) != m['ocr_manifest_sha256']:
            raise ValueError('Baseline OCR manifest changed')
        records = read(path)['documents']
        for doc, filename in SOURCES.items():
            record, = [r for r in records if r['source'] == filename]
            if verify == 'paddle':
                verify_run(record, ROOT / 'data' / filename, path.parent, PAGE_COUNTS[doc])
            else:
                responses, _ = verify_openai(record, path.parent, doc)
                verified_text[doc]['gpt-6-astra'] = {
                    'pages': ['\n\n'.join(r['markdown'] for r in responses)], 'input_format': 'markdown'}
    profiles, refs, predictions = [read(BASE / name) for name in ['profiles.json', 'references.json', 'predictions.json']]
    fields = read(BASE / 'critical_fields.json')
    for key, value in [('profile_sha256', profiles), ('reference_sha256', refs), ('predictions_sha256', predictions)]:
        if fingerprint(value) != fields['settings'][key]:
            raise ValueError('Frozen pilot input changed')
    data = read(BASE / 'input.json')
    for doc, methods in data.items():
        for method, record in methods.items():
            if record != verified_text[doc][method]:
                raise ValueError('Comparison text differs from verified baseline artifacts')
    if require_current_extractor and fields['extractor_version'] != EXTRACTOR_VERSION:
        raise ValueError('Shared extractor version changed')
    for doc, methods in (data.items() if require_current_extractor else []):
        for method, record in methods.items():
            # Historical pilot used Markdown decoding for the six saved labels.
            fmt = 'markdown' if method in FAMILIES else record.get('input_format', 'markdown')
            text = field_text('\n\n'.join(record['pages']), fmt)
            if extract(doc, text) != predictions['methods'][method]['documents'][doc]['fields']:
                raise ValueError('Shared extractor does not reproduce baseline')
    return manifest, data, profiles, refs, predictions, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ocr-output', type=Path, default=ROOT / 'output/unlimited-ocr-bf16')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/unlimited_ocr_experiment')
    parser.add_argument('--compare-ocr-output', type=Path, help='Verify and compare a previous Unlimited-OCR run')
    parser.add_argument('--compare-experiment', type=Path, help='Previous experiment with the same frozen peers/references')
    args = parser.parse_args()
    if bool(args.compare_ocr_output) != bool(args.compare_experiment):
        parser.error('Supply both --compare-ocr-output and --compare-experiment')
    root, out = args.ocr_output.resolve(), args.output.resolve()
    if args.compare_experiment and out == args.compare_experiment.resolve():
        parser.error('Use a separate output directory to preserve the previous experiment')
    manifest_path = root / 'manifest.json'
    manifest = read(manifest_path)
    if manifest['failures'] or len(manifest['documents']) != len(SOURCES):
        raise ValueError('Require a complete three-document Unlimited-OCR run')
    baseline_manifest, data, profiles, refs, predictions, baseline_fields = verified_baselines()
    for doc in data:
        for method in EXCLUDED:
            data[doc].pop(method, None)
    for method in EXCLUDED:
        predictions['methods'].pop(method, None)
    out.mkdir(parents=True, exist_ok=True)
    configs, timing, docs, response_docs = [], [], {}, {}
    for doc, filename in SOURCES.items():
        source = ROOT / 'data' / filename
        if sha(source) != SOURCE_SHA256[doc]:
            raise ValueError('Source differs from frozen pilot')
        record, = [r for r in manifest['documents'] if r['source'] == filename]
        responses, config = verify_unlimited(record, root, source, PAGE_COUNTS[doc])
        response_docs[doc] = responses
        configs.append(config)
        markdown = '\n\n'.join(r['markdown'] for r in responses)
        data[doc][METHOD] = {'pages': [markdown], 'input_format': 'markdown'}
        text = field_text(markdown, 'markdown')
        (out / (doc + '-decoded.txt')).write_text(text, encoding='utf-8')
        docs[doc] = {'fields': extract(doc, text)}
        timing.extend({'document': doc, 'page': r['page'], 'seconds': r['seconds'],
            'generation_tokens': r['generation_tokens'], 'finish_reason': r['finish_reason'],
            'rotation_ccw_degrees': r.get('rotation_ccw_degrees', 0)} for r in responses)
    if any(c != configs[0] for c in configs):
        raise ValueError('Documents use different configurations')
    predictions['methods'][METHOD] = {'family': 'deepseek', 'documents': docs,
        'provenance': {'manifest_sha256': sha(manifest_path), 'config': configs[0]}}
    fields = evaluate_benchmark(profiles, refs, predictions, split='validation')
    fields.update(meaning=baseline_fields['meaning'], extractor_version=EXTRACTOR_VERSION,
                  pilot_limitations=baseline_fields['pilot_limitations'])
    for method in baseline_fields['methods'].keys() - EXCLUDED:
        if fields['methods'][method]['metrics'] != baseline_fields['methods'][method]['metrics']:
            raise ValueError('Baseline pilot scores changed')
    print('Computing whole-document token agreement...', flush=True)
    result, pairs, volumes, normalized = evaluate(data, ontology_path=None, overlap_weight=1.0, weighting='document')
    families = {m: f for m, f in baseline_manifest['families'].items() if m not in EXCLUDED}
    families[METHOD] = 'deepseek'  # Conservative grouping: related DeepSeek-derived architecture.
    sensitivity = consensus_family_sensitivity(pairs, families)
    result.update(comparison_unit='whole_document', excluded_methods=sorted(EXCLUDED))
    for name, value in [('input', data), ('profiles', profiles), ('references', refs), ('predictions', predictions),
        ('critical_fields', fields), ('results', result), ('pair_differences', pairs), ('text_volume', volumes),
        ('normalized', normalized), ('family_sensitivity', sensitivity)]:
        write_json(out / (name + '.json'), value)
    audit = [{'method': method, 'document': doc, 'field': f['field'],
        'source_page': f['reference_location']['page'], 'reference': f['reference_value'],
        'prediction': f['predicted_value'], 'result': f['reason']}
        for method, scored in fields['methods'].items() for doc, scored_doc in scored['documents'].items()
        for f in scored_doc['field_results']]
    for name, rows in [('ranking', result['ranking']), ('critical_field_ranking', fields['ranking']),
                       ('field_audit', audit), ('page_timing', timing)]:
        write_csv(out / (name + '.csv'), rows)
    evaluation_manifest = {'ocr_manifest_sha256': sha(manifest_path),
        'baseline_manifest_sha256': sha(BASE / 'run_manifest.json'), 'source_sha256': SOURCE_SHA256,
        'input_sha256': sha(out / 'input.json'), 'families': families, 'extractor_version': EXTRACTOR_VERSION,
        'evaluation_script_sha256': sha(Path(__file__)),
        'runner_script_sha256': configs[0]['runner_sha256']}
    comparison = None
    if args.compare_ocr_output:
        comparison = compare_previous(args.compare_ocr_output.resolve(), args.compare_experiment.resolve(),
            out, data, profiles, refs, fields, result, response_docs, configs[0])
        evaluation_manifest['comparison'] = comparison['provenance']
    write_json(out / 'run_manifest.json', evaluation_manifest)
    new = fields['methods'][METHOD]
    lines = ['# Unlimited-OCR rotation-correction experiment' if configs[0].get('orientation')
             else '# Unlimited-OCR additional experiment', '',
        'Completed all 48 pages of the same three PDFs locally using the pinned BF16 MLX community conversion. '
        'This is single-page inference on Apple Silicon, not Baidu’s CUDA runtime or one-shot multi-page inference.', '',
        f'Model: `{MODEL}@{REVISION}`; MLX-VLM {configs[0]["runtime_versions"]["mlx-vlm"]}. '
        '144 DPI, gundam (1024 global + 640 local crops), greedy decoding, 32768 output tokens/page, '
        'sliding no-repeat n-gram size 35/window 128. No benchmark values enter inference.', '',
        '## Whole-document token agreement without ontology', '',
        'Equal document weighting, token overlap only. Scores measure peer agreement, not reference accuracy. '
        'Historical Tesseract-labelled openai-6 and fable-5-1 are excluded. DeepSeek and Unlimited-OCR '
        'share a conservative family group in sensitivity analysis. Scores depend on this peer set.', '',
        '| Rank | Method | Agreement / 100 | CSA | Board | Trade | Family-balanced |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for row in result['ranking']:
        per_doc = [next(r['consensus_score'] for r in result['per_document_rankings'][doc]
                        if r['model'] == row['model']) for doc in ['csa', 'board', 'trade']]
        lines.append(f'| {row["rank"]} | {row["model"]} | {row["consensus_score"]:.2f} | '
                     + ' | '.join(f'{v:.2f}' for v in per_doc)
                     + f' | {sensitivity["family_balanced_scores"][row["model"]]:.2f} |')
    lines += ['', '## Frozen critical-field pilot', '',
        f'Unlimited-OCR passes {new["metrics"]["critical_correct"]}/27 selected fields; '
        f'equal-document score {new["score"]:.2f}/100. Shared extractor version {EXTRACTOR_VERSION}; '
        'references and scoring policies unchanged. This measures OCR plus extraction on a convenience sample; '
        'it is neither held-out nor independently adjudicated full-corpus accuracy.', '',
        '| Method | Score / 100 | Correct / 27 | Board / 13 | CSA / 6 | Trade / 8 |',
        '|---|---:|---:|---:|---:|---:|']
    for rank in fields['ranking']:
        scored = fields['methods'][rank['method']]
        counts = [scored['documents'][d]['counts']['critical_correct'] for d in ['board', 'csa', 'trade']]
        lines.append(f'| {rank["method"]} | {scored["score"]:.2f} | {scored["metrics"]["critical_correct"]} | '
                     + ' | '.join(map(str, counts)) + ' |')
    orientation = configs[0].get('orientation')
    if orientation:
        changed = [(doc, i, a) for doc, record in orientation['documents'].items()
                   for i, a in enumerate(record['angles_ccw'], 1) if a]
        lines += ['', '## Rotation correction', '',
            f'Used only page-angle predictions from the verified saved `{orientation["classifier"]}` classifier. '
            f'{len(changed)} pages were transposed losslessly; all other images were unchanged. '
            'PaddleOCR transcription text was not supplied to Unlimited-OCR. Document angles were reused '
            'from saved checkpoints, so run time excludes orientation classifier inference.', '',
            *[f'- {name}, page {page}: {angle} degrees counterclockwise.' for name, page, angle in changed]]
    if comparison:
        lines += ['', '## Before/after on unchanged peers and references', '',
            f'Whole-text agreement: {comparison["before"]["agreement"]:.2f} → {comparison["after"]["agreement"]:.2f}. '
            f'Pilot: {comparison["before"]["correct"]}/27 → {comparison["after"]["correct"]}/27; '
            f'equal-document score {comparison["before"]["score"]:.2f} → {comparison["after"]["score"]:.2f}. '
            f'Recovered {comparison["recovered_missing"]} of the {comparison["previous_missing"]} previously missing fields. '
            f'Exact reference literals became present on the annotated source page for '
            f'{comparison["raw_literal_recovered"]} previously missing fields. '
            'This is a post-hoc rotation ablation on the same convenience sample.', '',
            'Literal presence is a transcription diagnostic, not association/extraction score credit. '
            'Invoice number/date labels may be emitted separately from their values.', '',
            '| Document | Previously missing field | Before | After extraction | Literal in raw page | Extracted value |',
            '|---|---|---|---|---|---|']
        lines += [f'| {r["document"]} | {r["field"]} | {r["before_reason"]} | {r["after_reason"]} | '
                  f'{r["after_native_literal_present"]} | '
                  f'{str(r["after_value"] or "").replace("|", "/")} |' for r in comparison['recovery_audit']]
        lines += ['', f'{comparison["unrotated_text_changed"]} unrotated pages changed raw output text on rerun. '
                  'See page_changes.csv for all 48 image/text comparisons.']
    failures = [(doc, f['field'], f['reason']) for doc, scored in new['documents'].items()
                for f in scored['field_results'] if f['reason'] != 'correct']
    lines += ['', '## Observed failures', '',
        '; '.join(f'{d} {field}: {reason}' for d, field, reason in failures) if failures else 'No selected pilot field failures.', '',
        'Normal stop status indicates generation completion, not correct or complete transcription. '
        'No value repair or extractor changes were applied.', '',
        '## Runtime and provenance', '',
        f'Summed page inference time: {sum(r["seconds"] for r in timing):.2f}s. '
        f'Run wall time including load/render/save: {manifest["wall_seconds"]:.2f}s. '
        f'Generated tokens: {sum(r["generation_tokens"] for r in timing):,}. '
        'Every page finished with stop status. Source bytes, rendered pixels, model files, settings and artifacts are hashed.', '',
        'Baseline caveats: Insavlo identity/source provenance and historical Fable provenance are unverified; '
        'Mistral trade used an earlier TIFF, and saved DeepSeek trade page 10 was truncated.', '',
        'See field_audit.csv for all field failures and page_timing.csv for page timings.', '',
        '## Reproduce', '', '```bash',
        'conda run --no-capture-output -n myenv3.13 python scripts/run_unlimited_ocr.py' +
        (' --orientation-manifest output/paddleocr-v4/manifest.json' if orientation else ''),
        'conda run --no-capture-output -n myenv3.13 python scripts/evaluate_unlimited_ocr.py' +
        f' --ocr-output {command_path(root)} --output {command_path(out)}' +
        (f' --compare-ocr-output {command_path(args.compare_ocr_output)}'
         f' --compare-experiment {command_path(args.compare_experiment)}' if comparison else ''), '```', '',
        '[Baidu model](https://huggingface.co/baidu/Unlimited-OCR); '
        '[MLX BF16 conversion](https://huggingface.co/mlx-community/Unlimited-OCR-bf16).', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'agreement': result['ranking'], 'pilot': fields['ranking']}, indent=2))
    print(out / 'report.md')


def compare_previous(ocr_root, experiment, output, data, profiles, refs, fields, result, current_pages, config):
    """Require verified artifacts and identical peers; preserve original scores."""
    previous_manifest_path = ocr_root / 'manifest.json'
    previous_manifest = read(previous_manifest_path)
    previous_eval = read(experiment / 'run_manifest.json')
    if (previous_manifest['failures'] or len(previous_manifest['documents']) != len(SOURCES) or
            sha(previous_manifest_path) != previous_eval['ocr_manifest_sha256'] or
            sha(experiment / 'input.json') != previous_eval['input_sha256']):
        raise ValueError('Previous experiment provenance mismatch')
    old_data, old_fields = read(experiment / 'input.json'), read(experiment / 'critical_fields.json')
    if (fingerprint(profiles) != old_fields['settings']['profile_sha256'] or
            fingerprint(refs) != old_fields['settings']['reference_sha256'] or
            old_fields['extractor_version'] != EXTRACTOR_VERSION):
        raise ValueError('Before/after uses different pilot references or extractor')
    page_changes, previous_documents, previous_pages = [], {}, {}
    for doc, name in SOURCES.items():
        for method, record in data[doc].items():
            if method != METHOD and record != old_data[doc][method]:
                raise ValueError('Before/after peers changed')
        if set(data[doc]) != set(old_data[doc]):
            raise ValueError('Before/after peer sets differ')
        record, = [r for r in previous_manifest['documents'] if r['source'] == name]
        responses, old_config = verify_unlimited(record, ocr_root, ROOT / 'data' / name, PAGE_COUNTS[doc])
        previous_pages[doc] = responses
        ignore = {'runner_sha256', 'orientation'}
        if {k: v for k, v in config.items() if k not in ignore} != {k: v for k, v in old_config.items() if k not in ignore}:
            raise ValueError('Inference settings changed beyond rotation')
        markdown = '\n\n'.join(r['markdown'] for r in responses)
        if old_data[doc][METHOD] != {'pages': [markdown], 'input_format': 'markdown'}:
            raise ValueError('Previous comparison text differs from native artifacts')
        previous_documents[doc] = {'fields': extract(doc, field_text(markdown, 'markdown'))}
        for old, new in zip(responses, current_pages[doc]):
            angle = new.get('rotation_ccw_degrees', 0)
            if old['image_sha256'] != new.get('original_image_sha256', new['image_sha256']):
                raise ValueError('Before/after original rendered pixels differ')
            page_changes.append({'document': doc, 'page': new['page'], 'rotation_ccw_degrees': angle,
                'image_changed': old['image_sha256'] != new['image_sha256'],
                'raw_text_changed': old['text'] != new['text'], 'before_tokens': old['generation_tokens'],
                'after_tokens': new['generation_tokens']})
    predictions = read(experiment / 'predictions.json')
    if fingerprint(predictions) != old_fields['settings']['predictions_sha256']:
        raise ValueError('Previous predictions changed')
    if previous_documents != predictions['methods'][METHOD]['documents']:
        raise ValueError('Previous shared extraction differs')
    recomputed = evaluate_benchmark(profiles, refs, predictions, split='validation')
    old_score = recomputed['methods'][METHOD]
    if old_score['metrics'] != old_fields['methods'][METHOD]['metrics']:
        raise ValueError('Previous field scores changed')
    old_result, _, _, _ = evaluate(old_data, ontology_path=None, overlap_weight=1.0, weighting='document')
    saved_result = read(experiment / 'results.json')
    if old_result['ranking'] != saved_result['ranking']:
        raise ValueError('Previous agreement scores changed')
    new_score = fields['methods'][METHOD]
    recovery = []
    for doc, scored in old_score['documents'].items():
        now = {f['field']: f for f in new_score['documents'][doc]['field_results']}
        for f in scored['field_results']:
            if f['reason'] == 'missing':
                after = now[f['field']]
                page = f['reference_location']['page']
                reference = str(f['reference_value'])
                recovery.append({'document': doc, 'field': f['field'], 'before_reason': f['reason'],
                    'after_reason': after['reason'], 'after_value': after['predicted_value'],
                    'reference': f['reference_value'], 'source_page': page,
                    'before_native_literal_present': reference in previous_pages[doc][page - 1]['text'],
                    'after_native_literal_present': reference in current_pages[doc][page - 1]['text']})
    before_agreement = next(r['consensus_score'] for r in old_result['ranking'] if r['model'] == METHOD)
    after_agreement = next(r['consensus_score'] for r in result['ranking'] if r['model'] == METHOD)
    comparison = {'before': {'agreement': before_agreement, 'score': old_score['score'],
        'correct': old_score['metrics']['critical_correct']},
        'after': {'agreement': after_agreement, 'score': new_score['score'],
                  'correct': new_score['metrics']['critical_correct']},
        'previous_missing': len(recovery), 'recovered_missing': sum(r['after_reason'] == 'correct' for r in recovery),
        'raw_literal_recovered': sum(not r['before_native_literal_present'] and r['after_native_literal_present'] for r in recovery),
        'recovery_audit': recovery, 'unrotated_text_changed': sum(not p['rotation_ccw_degrees'] and p['raw_text_changed'] for p in page_changes),
        'provenance': {'previous_ocr_manifest': str(previous_manifest_path), 'sha256': sha(previous_manifest_path),
            'previous_experiment_manifest_sha256': sha(experiment / 'run_manifest.json'),
            'changed_image_pages': sum(p['image_changed'] for p in page_changes)}}
    write_csv(output / 'page_changes.csv', page_changes)
    write_csv(output / 'recovery_audit.csv', recovery)
    write_json(output / 'change_comparison.json', comparison)
    return comparison


if __name__ == '__main__':
    main()
