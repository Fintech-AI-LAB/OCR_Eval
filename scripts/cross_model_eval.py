#!/usr/bin/env python3
"""Evaluate page-aligned OCR text without ground truth; uses only the standard library."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

from evaluate_outputs import ALGORITHM_VERSION, normalize_texts, tokens, compare_page, load
from ocr_common import ROOT, write_json


def validate(data):
    if not isinstance(data, dict) or not data:
        raise ValueError('Input must be a nonempty mapping of documents to models.')
    models = None
    for document, records in data.items():
        if not isinstance(document, str) or not isinstance(records, dict):
            raise ValueError('Each document must map model names to page records.')
        if models is None:
            models = sorted(records)
            if len(models) < 3:
                raise ValueError('Consensus ranking requires at least three models.')
        if set(records) != set(models):
            raise ValueError(f'{document}: model coverage differs; align inputs explicitly.')
        counts = set()
        for model, record in records.items():
            pages = record.get('pages') if isinstance(record, dict) else None
            if not isinstance(pages, list) or not pages or not all(isinstance(p, str) for p in pages):
                raise ValueError(f'{document}/{model}: pages must be a nonempty list of strings.')
            counts.add(len(pages))
        if len(counts) != 1:
            raise ValueError(f'{document}: page counts differ; do not silently zip unequal lists.')
    return models


def rank(scores):
    """Competition ranks; numerical ties receive the same rank, sorted by name for display."""
    result = []
    previous = None
    for position, model in enumerate(sorted(scores, key=lambda m: (-scores[m], m)), 1):
        value = scores[model]
        if previous is None or not math.isclose(value, previous, rel_tol=0, abs_tol=1e-12):
            current_rank = position
        result.append({'rank': current_rank, 'model': model, 'consensus_score': 100 * value})
        previous = value
    return result


def evaluate(data, overlap_weight=1.0, weighting='document'):
    if not math.isfinite(overlap_weight) or not 0 <= overlap_weight <= 1:
        raise ValueError('overlap_weight must be finite and between 0 and 1.')
    if weighting not in ('document', 'page'):
        raise ValueError('weighting must be document or page.')
    models = validate(data)
    normalized = {d: {m: {'pages': [], 'input_format': 'plain'} for m in records} for d, records in data.items()}
    for document, records in data.items():
        for index in range(len(records[models[0]]['pages'])):
            texts = normalize_texts([records[m]['pages'][index] for m in models],
                                    [records[m].get('input_format', 'markdown') for m in models])
            for model, text in zip(models, texts):
                normalized[document][model]['pages'].append(text)
    comparisons, volumes, excluded = [], [], []
    doc_scores = {}
    page_scores = {m: [] for m in models}
    for document, records in normalized.items():
        per_model = {m: [] for m in models}
        for index in range(len(records[models[0]]['pages'])):
            texts = {m: records[m]['pages'][index] for m in models}
            for m, text in texts.items():
                volumes.append({'document': document, 'page': index + 1,
                                'model': m, 'characters': len(text)})
            if not any(tokens(t) for t in texts.values()):
                excluded.append({'document': document, 'page': index + 1, 'reason': 'all texts lack scorable tokens'})
                continue
            edges = {m: [] for m in models}
            for a, b in itertools.combinations(models, 2):
                row = compare_page(document, index + 1, a, b, texts[a], texts[b])
                # Empty-empty is not positive corroboration when others contain text.
                if not tokens(texts[a]) and not tokens(texts[b]):
                    row['token_overlap'] = row['sequence_similarity'] = 0.0
                row['combined_similarity'] = (overlap_weight * row['token_overlap']
                                              + (1 - overlap_weight) * row['sequence_similarity'])
                row['primary_similarity'] = row['combined_similarity']
                comparisons.append(row)
                edges[a].append(row['combined_similarity'])
                edges[b].append(row['combined_similarity'])
            for model in models:
                score = statistics.mean(edges[model])
                per_model[model].append(score)
                page_scores[model].append(score)
        if per_model[models[0]]:
            doc_scores[document] = {m: statistics.mean(v) for m, v in per_model.items()}
    if not doc_scores:
        raise ValueError('All page texts are empty; no evidence for ranking.')
    overall = {m: statistics.mean(scores[m] for scores in doc_scores.values())
               if weighting == 'document' else statistics.mean(page_scores[m]) for m in models}
    summary = []
    for a, b in itertools.combinations(models, 2):
        rows = [r for r in comparisons if r['left'] == a and r['right'] == b]
        numeric = [r['numeric_agreement'] for r in rows if r['numeric_agreement'] is not None]
        summary.append({'left': a, 'right': b, 'pages': len(rows),
                        'pages_with_numbers': len(numeric),
                        'mean_numeric_agreement': statistics.mean(numeric) if numeric else None,
                        'mean_page_token_overlap': statistics.mean(r['token_overlap'] for r in rows),
                        'mean_page_sequence_similarity': statistics.mean(r['sequence_similarity'] for r in rows),
                        'mean_page_combined_similarity': statistics.mean(r['combined_similarity'] for r in rows)})
    return {'schema_version': 3, 'algorithm_version': ALGORITHM_VERSION, 'meaning': 'Text consensus, not OCR accuracy',
            'settings': {'primary_metric': 'token_count_overlap' if overlap_weight == 1 else 'custom_blend',
                         'overlap_weight': overlap_weight, 'sequence_weight': 1 - overlap_weight,
                         'document_weighting': weighting, 'empty_pair_policy': 'zero when other methods have text'},
            'models': models, 'documents': list(normalized), 'ranking': rank(overall),
            'per_document_rankings': {d: rank(v) for d, v in doc_scores.items()},
            'excluded_pages': excluded, 'pairwise_summary': summary}, comparisons, volumes, normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='JSON: document -> model -> {pages: [text, ...]}; default reads repository OCR artifacts')
    parser.add_argument('--models', nargs='+', help='Evaluate only these model labels')
    parser.add_argument('--documents', nargs='+', help='Evaluate only these document IDs')
    parser.add_argument('--overlap-weight', type=float, default=1.0,
                        help='Default 1: ignore reading order. Values below 1 opt into sequence scoring.')
    parser.add_argument('--weighting', choices=['document', 'page'], default='document')
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'cross_model')
    args = parser.parse_args()
    try:
        if args.input:
            raw = args.input.read_bytes()
            data = json.loads(raw)
            provenance = {'input_file': str(args.input.resolve()),
                          'input_sha256': hashlib.sha256(raw).hexdigest()}
        else:
            data, provenance = load(normalize=False)
        if args.documents:
            if len(args.documents) != len(set(args.documents)):
                raise ValueError('Duplicate document selection.')
            data = {d: data[d] for d in args.documents}
        if args.models:
            if len(args.models) != len(set(args.models)):
                raise ValueError('Duplicate model selection.')
            data = {d: {m: records[m] for m in args.models} for d, records in data.items()}
        result, pairs, volumes, normalized = evaluate(data, args.overlap_weight, args.weighting)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))
    result['provenance'] = provenance
    args.output.mkdir(parents=True, exist_ok=True)
    for name, obj in [('results.json', result), ('page_differences.json', pairs),
                      ('text_volume.json', volumes), ('normalized_pages.json', normalized)]:
        write_json(args.output / name, obj)
    with (args.output / 'ranking.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['rank', 'model', 'consensus_score'])
        writer.writeheader(); writer.writerows(result['ranking'])
    lines = ['# Cross-model text evaluation', '',
             f"Documents: {len(result['documents'])}; methods: {len(result['models'])}; excluded all-empty pages: {len(result['excluded_pages'])}.", '',
             '**Scores measure consensus, not correctness.** No ground-truth model or structure score is used.', '',
             f"Pair score = {args.overlap_weight:.2f} × token overlap + {1-args.overlap_weight:.2f} × symmetric sequence similarity. Average across peers, then use equal {args.weighting} weighting. Scoring ignores formatting and case, but preserves signed numbers, numeric separators, percentages and leading zeroes. Numeric agreement is reported separately. Line hyphens are joined only when another output corroborates the joined word.", '',
             '| Rank | Saved model label | Consensus / 100 |', '|---:|---|---:|']
    for row in result['ranking']:
        lines.append(f"| {row['rank']} | {row['model'].replace('|', '/')} | {row['consensus_score']:.2f} |")
    lines += ['', 'Empty text is not evidence of a correct blank page. All-empty pages are excluded; two empty methods receive zero agreement when another method has text. Ties share ranks.', '',
              'Model labels are supplied by the input. Shared errors and related OCR pipelines can inflate agreement. The repository openai-6 and fable-5-1 artifacts identify Tesseract with visual review; their labels do not verify those named models. Fable raw page text is used, not separately reviewed fields.', '',
              'See results.json for per-document rankings, pairwise summaries and provenance; page_differences.json for differing passages and numeric strings. No original OCR files were modified and no inference calls were made.']
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for row in result['ranking']:
        print(f"{row['rank']}. {row['model']}: {row['consensus_score']:.2f}")
    print(f'Results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
