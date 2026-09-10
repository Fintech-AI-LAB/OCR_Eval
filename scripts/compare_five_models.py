"""Compare five saved OCR methods using whole-document text agreement."""
import csv
import hashlib
import json
import statistics

from cross_model_eval import evaluate
from evaluate_outputs import ALGORITHM_VERSION, DOCS, load
from ocr_common import ROOT, write_json


def main():
    data, provenance = load(normalize=False)
    output = ROOT / 'output' / 'five_model_comparison'
    insavlo_paths = {
        'csa': ROOT / 'output/insavlo' / (DOCS['csa'] + '.md'),
        'board': ROOT / 'output/insavlo/board_resolution.md',
        'trade': ROOT / 'output/insavlo/Trade-1118348-260625.001369.01.01.tif0.tiff.md',
    }
    source_page_counts = {}
    for document, methods in data.items():
        source_page_counts[document] = {m: len(record['pages']) for m, record in methods.items()}
        for record in methods.values():
            record['pages'] = ['\n\n'.join(record['pages'])]
        path = insavlo_paths[document]
        raw = path.read_bytes()
        methods['insavlo'] = {'pages': [raw.decode('utf-8-sig')]}
        provenance.append({'document': document, 'engine_label': 'insavlo',
                           'artifacts': [str(path)], 'sha256': hashlib.sha256(raw).hexdigest(),
                           'source_hash_verified': None,
                           'metadata': 'Markdown only; model provenance and physical page boundaries unavailable.'})
    result, pairs, volumes, normalized = evaluate(data, overlap_weight=1.0, weighting='document')
    # The evaluator uses one text unit per record: here each unit is a DOCUMENT.
    for row in pairs + volumes:
        row.pop('page', None)
        row['unit'] = 'whole_document'
    for row in result['pairwise_summary']:
        row['documents'] = row.pop('pages')
        row['documents_with_numbers'] = row.pop('pages_with_numbers')
        for key in list(row):
            if key.startswith('mean_page_'):
                row[key.replace('mean_page_', 'mean_document_')] = row.pop(key)
    result['settings']['comparison_unit'] = 'whole_document'
    result['settings']['aggregation'] = 'Equal weight for each of three documents; not a page average.'
    result['source_page_counts'] = source_page_counts
    result['excluded_documents'] = result.pop('excluded_pages')
    result['provenance'] = provenance
    result['overall_agreement_score'] = statistics.mean(r['consensus_score'] for r in result['ranking'])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', result)
    write_json(output / 'document_differences.json', pairs)
    write_json(output / 'text_volume.json', volumes)
    write_json(output / 'normalized_documents.json', {
        d: {m: r['pages'][0] for m, r in methods.items()} for d, methods in normalized.items()})
    with (output / 'ranking.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['rank', 'model', 'consensus_score'])
        writer.writeheader()
        writer.writerows(result['ranking'])
    lines = ['# Five-method text consensus comparison', '',
             'Three complete documents; each document has equal weight. Insavlo lacks consistent physical page boundaries, so this is a whole-document comparison, not a page-average comparison. All methods use the same document scope.', '',
             f'Algorithm version {ALGORITHM_VERSION}: primary agreement is token-count overlap, independent of reading order. Each method is scored against its four peers, then averaged across documents. Formatting and case are ignored; signed numbers, decimal separators, accounting parentheses and percentages are retained. Sequence similarity and numeric agreement are separate diagnostics and do not contribute additional weight to the ranking. Numeric agreement counts repeated occurrences. Numeric tokens are included in token overlap. No structure or layout score is used. A reordered text can receive 100 even when word associations differ; this is a content-consensus measure.', '',
             '**Consensus is not accuracy.** Shared errors can increase scores. The openai-6 and fable-5-1 artifacts describe Tesseract with visual review; their folder labels do not verify the named models. Insavlo has no model provenance metadata. No OCR was rerun.', '',
             '| Rank | Method | Mean consensus / 100 | CSA | Board | Trade |',
             '|---:|---|---:|---:|---:|---:|']
    for row in result['ranking']:
        per_doc = [next(r['consensus_score'] for r in result['per_document_rankings'][d]
                        if r['model'] == row['model']) for d in ('csa', 'board', 'trade')]
        lines.append(f"| {row['rank']} | {row['model']} | {row['consensus_score']:.2f} | " +
                     ' | '.join(f'{score:.2f}' for score in per_doc) + ' |')
    lines += ['', f"Overall agreement across all methods and documents: {result['overall_agreement_score']:.2f}/100.", '',
              'These scores are not directly comparable to the earlier four-method page-based scores: both the peer set and comparison unit changed.', '',
              'Files: results.json includes per-document rankings, pairwise summaries and provenance; document_differences.json contains text and numeric disagreements; ranking.csv contains the overall ranking.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result['ranking'], indent=2))
    print(output / 'report.md')


if __name__ == '__main__':
    main()
