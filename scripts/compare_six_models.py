"""Compare six saved OCR outputs using equal-weight whole-document consensus."""
import csv
import hashlib
import json
import statistics

from cross_model_eval import evaluate
from evaluate_outputs import ALGORITHM_VERSION, DOCS, load
from ocr_common import ROOT, write_json


def main():
    data, provenance = load(normalize=False)
    output = ROOT / 'output/six_model_comparison'
    manifest_path = ROOT / 'output/deepseek-ocr2/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    counts, warnings = {}, []
    for document, methods in data.items():
        counts[document] = {m: len(r['pages']) for m, r in methods.items()}
        expected = next(iter(counts[document].values()))
        if set(counts[document].values()) != {expected}:
            raise ValueError(f'{document}: existing methods disagree on page count')
        insavlo = ROOT / 'output/insavlo' / (DOCS[document] + ('.tiff.md' if document == 'trade' else '.md'))
        raw = insavlo.read_bytes()
        methods['insavlo'] = {'pages': [raw.decode('utf-8-sig')], 'input_format': 'markdown'}
        counts[document]['insavlo'] = None
        provenance.append({'document': document, 'engine_label': 'insavlo',
                           'artifacts': [str(insavlo)], 'artifact_sha256': {str(insavlo): hashlib.sha256(raw).hexdigest()},
                           'source_hash_verified': None,
                           'metadata': 'Document Markdown; physical page boundaries and model identity unverified.'})
        selected = [r for r in manifest['documents'] if r['source'] == DOCS[document] + '.pdf']
        if len(selected) != 1:
            raise ValueError(f'{document}: expected one selected DeepSeek run')
        record = selected[0]
        folder = ROOT / 'output/deepseek-ocr2' / record['source'] / record['run']
        metadata_path = folder / 'source.json'
        metadata = json.loads(metadata_path.read_text())
        paths = sorted(folder.glob('response-*.json'))
        if len(paths) != expected or metadata['responses'] != expected or record['pages'] != expected:
            raise ValueError(f'{document}: DeepSeek page-count mismatch')
        source = ROOT / 'data' / record['source']
        config = {k: v for k, v in metadata.items() if k not in ('source', 'responses', 'status', 'truncated_pages')}
        digest = hashlib.sha256(source.read_bytes())
        digest.update(json.dumps(config, sort_keys=True).encode())
        if digest.hexdigest()[:16] != record['run']:
            raise ValueError(f'{document}: DeepSeek source/settings hash mismatch')
        texts, capped = [], []
        for index, path in enumerate(paths, 1):
            response = json.loads(path.read_text())
            if response['page'] != index or path.name != f'response-{index:04d}.json':
                raise ValueError(f'{document}: DeepSeek page order mismatch')
            texts.append(response['markdown'])  # No runner-added warning text in scoring.
            if response.get('truncated'):
                capped.append(index)
        if capped != record['truncated_pages']:
            raise ValueError(f'{document}: truncation metadata mismatch')
        if capped:
            warnings.append({'document': document, 'model': 'deepseek', 'truncated_pages': capped,
                             'policy': 'Include saved model output, including repetitions; exclude runner warning text.'})
        methods['deepseek'] = {'pages': texts, 'input_format': 'markdown'}
        counts[document]['deepseek'] = len(texts)
        artifacts = [metadata_path, *paths]
        provenance.append({'document': document, 'engine_label': 'deepseek', 'metadata': metadata,
                           'source_pdf': str(source), 'source_hash_verified': True,
                           'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                           'artifacts': list(map(str, artifacts)),
                           'artifact_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}})
        for method in methods.values():
            method['pages'] = ['\n\n'.join(method['pages'])]

    result, pairs, volumes, normalized = evaluate(data, overlap_weight=1.0, weighting='document')
    for row in pairs + volumes:
        row.pop('page', None)
        row['unit'] = 'whole_document'
    for row in result['pairwise_summary']:
        row['documents'] = row.pop('pages')
        row['documents_with_numbers'] = row.pop('pages_with_numbers')
        row['documents_with_regions'] = row.pop('pages_with_regions')
        row['documents_with_field_values'] = row.pop('pages_with_field_values')
        for key in list(row):
            if key.startswith('mean_page_'):
                row[key.replace('mean_page_', 'mean_document_')] = row.pop(key)
    result['excluded_documents'] = result.pop('excluded_pages')
    result['settings'].update(comparison_unit='whole_document', aggregation='Equal document weight; five peers per method')
    result.update(source_page_counts=counts, provenance=provenance, warnings=warnings,
                  deepseek_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=True)
    for filename, value in [('results.json', result), ('document_differences.json', pairs),
                            ('text_volume.json', volumes), ('normalized_documents.json',
                             {d: {m: r['pages'][0] for m, r in methods.items()} for d, methods in normalized.items()})]:
        write_json(output / filename, value)
    with (output / 'ranking.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['rank', 'model', 'consensus_score'])
        writer.writeheader()
        writer.writerows(result['ranking'])
    lines = ['# Six-method OCR comparison', '',
             '**Text consensus, not accuracy.** Three documents (14 CSA, 4 board, 30 trade pages); each has equal weight.', '',
             'Insavlo lacks reliable physical page boundaries, so all six methods are compared as whole documents, not page averages.', '',
             '| Rank | Method | Consensus / 100 | CSA | Board | Trade |',
             '|---:|---|---:|---:|---:|---:|']
    for row in result['ranking']:
        scores = [next((r['consensus_score'] for r in result['per_document_rankings'].get(d, []) if r['model'] == row['model']), None)
                  for d in ('csa', 'board', 'trade')]
        lines.append(f'| {row["rank"]} | {row["model"]} | {row["consensus_score"]:.2f} | ' +
                     ' | '.join(f'{s:.2f}' if s is not None else 'n/a' for s in scores) + ' |')
    lines += ['', '## Method', '',
              f'Algorithm {ALGORITHM_VERSION}: primary agreement compares all content in text regions around ontology labels. Each anchor has a window with 16 context tokens per side; overlapping token and bigram weight is shared so text counts once. One-to-one alignment requires the same central anchor tag and contextual evidence for ambiguous anchors. Token and ordered-bigram agreement are averaged and weighted by selected token mass; unmatched regions receive zero. Exact values and field owners are not inferred. Anchor agreement and selected-text coverage are separate diagnostics. Each method averages agreement with five peers, then across documents.', '',
              'Markdown/HTML formatting is decoded once. Signed numbers, numeric separators, percentages, leading zeroes and accounting parentheses are retained. Sequence and numeric agreement are separate diagnostics. No layout, speed or cost score is included.', '',
              '## Limitations', '',
              '- DeepSeek trade page 10 hit the 8192-token cap. Its saved text, including repetitions, is included; runner-added warning text is excluded. Dense trade tables also showed errors and omissions. The selected run comes from the DeepSeek manifest, not partial diagnostic runs.',
              '- The saved openai-6 and fable-5-1 artifacts describe Tesseract with visual review. Their labels do not establish inference by those named models. Insavlo model identity is unverified.',
              '- Mistral trade used an earlier TIFF; exact pixel equivalence to the current PDF is unverified. Fable and Insavlo lack source-hash verification.',
              '- Shared errors can inflate consensus. These scores do not establish correctness. Adding DeepSeek changes every method’s peer average, so scores differ from the five-method comparison.', '',
              '## Reproduce', '', '```bash',
              'conda run --no-capture-output -n myenv3.13 python scripts/compare_six_models.py', '```', '',
              'No OCR inference was run and no source outputs were changed. See results.json for provenance and pairwise summaries; document_differences.json for text/numeric differences; ranking.csv for the ranking.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result['ranking'], indent=2), flush=True)
    print(output / 'report.md', flush=True)


if __name__ == '__main__':
    main()
