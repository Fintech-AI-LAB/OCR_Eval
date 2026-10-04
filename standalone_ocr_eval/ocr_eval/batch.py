"""Evaluate page-aligned OCR text without ground truth; uses only the standard library."""
import itertools
import math
import statistics

from .text import ALGORITHM_VERSION, normalize_texts, tokens, compare_page
from .ontology import DEFAULT_ONTOLOGY, DEFAULT_ENTITY_WEIGHT, load_ontology, validate_entity_weight

from .field_values import extract_field_values, field_text, primary_metric, primary_overlap, validate_ontology_mode

from .regions import extract_ontology_regions, region_policy, validate_region_context, DEFAULT_REGION_CONTEXT_TOKENS


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


def evaluate(data, overlap_weight=1.0, weighting='document', ontology_path=DEFAULT_ONTOLOGY,
             entity_weight=DEFAULT_ENTITY_WEIGHT, ontology_mode='regions', region_context_tokens=DEFAULT_REGION_CONTEXT_TOKENS):
    validate_entity_weight(entity_weight)
    validate_ontology_mode(ontology_mode)
    validate_region_context(region_context_tokens)
    _, ontology_metadata = load_ontology(ontology_path)
    if not math.isfinite(overlap_weight) or not 0 <= overlap_weight <= 1:
        raise ValueError('overlap_weight must be finite and between 0 and 1.')
    if weighting not in ('document', 'page'):
        raise ValueError('weighting must be document or page.')
    models = validate(data)
    normalized = {d: {m: {'pages': [], 'input_format': 'plain'} for m in records} for d, records in data.items()}
    for document, records in data.items():
        for index in range(len(records[models[0]]['pages'])):
            sources = [field_text(records[m]['pages'][index], records[m].get('input_format', 'markdown')) for m in models]
            texts = normalize_texts(sources, ['plain'] * len(sources))
            for model, text, source in zip(models, texts, sources):
                normalized[document][model]['pages'].append(
                    source if ontology_path is not None and ontology_mode in ('values', 'regions') else text)
    comparisons, volumes, excluded, localizations = [], [], [], []
    doc_scores = {}
    page_scores = {m: [] for m in models}
    for document, records in normalized.items():
        per_model = {m: [] for m in models}
        for index in range(len(records[models[0]]['pages'])):
            sources = {m: records[m]['pages'][index] for m in models}
            texts = dict(zip(models, normalize_texts(list(sources.values()), ['plain'] * len(models))))
            fields = {m: extract_field_values(sources[m], ontology_path) if ontology_mode == 'values' else {'fields': [], 'issues': []} for m in models}
            regions = {m: extract_ontology_regions(sources[m], ontology_path if ontology_mode == 'regions' else None, context_tokens=region_context_tokens) for m in models}
            if ontology_path is not None and ontology_mode == 'regions':
                localizations.extend({'document': document, 'page': index + 1, 'model': m, **regions[m]['localization']} for m in models)
            for m, text in texts.items():
                volumes.append({'document': document, 'page': index + 1,
                                'model': m, 'characters': len(text)})
            if not any(tokens(t) for t in texts.values()):
                excluded.append({'document': document, 'page': index + 1, 'reason': 'all texts lack scorable tokens'})
                continue
            if ontology_path is not None and ontology_mode == 'values' and not any(f['fields'] or f['issues'] for f in fields.values()):
                excluded.append({'document': document, 'page': index + 1, 'reason': 'no ontology field values extracted'})
                continue
            if ontology_path is not None and ontology_mode == 'regions' and not any(r['regions'] for r in regions.values()):
                excluded.append({'document': document, 'page': index + 1, 'reason': 'no ontology regions located'})
                continue
            edges = {m: [] for m in models}
            for a, b in itertools.combinations(models, 2):
                row = compare_page(document, index + 1, a, b, texts[a], texts[b], ontology_path, entity_weight,
                                   left_fields=fields[a], right_fields=fields[b], left_regions=regions[a], right_regions=regions[b],
                                   ontology_mode=ontology_mode, region_context_tokens=region_context_tokens)
                # Empty-empty is not positive corroboration when others contain text.
                if not tokens(texts[a]) and not tokens(texts[b]):
                    row['token_overlap'] = row['sequence_similarity'] = 0.0
                row['combined_similarity'] = (overlap_weight * primary_overlap(row, ontology_path, ontology_mode)
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
        raise ValueError('No scorable pages; texts are empty or no ontology evidence was located. Check ontology labels or evaluate without an ontology.')
    overall = {m: statistics.mean(scores[m] for scores in doc_scores.values())
               if weighting == 'document' else statistics.mean(page_scores[m]) for m in models}
    summary = []
    for a, b in itertools.combinations(models, 2):
        rows = [r for r in comparisons if r['left'] == a and r['right'] == b]
        numeric = [r['numeric_agreement'] for r in rows if r['numeric_agreement'] is not None]
        summary.append({'left': a, 'right': b, 'pages': len(rows),
                        'pages_with_numbers': len(numeric),
                        'mean_numeric_agreement': statistics.mean(numeric) if numeric else None,
                        'pages_with_regions': sum(r['ontology_region_total_tokens'] > 0 for r in rows),
                        'mean_page_region_agreement': statistics.mean(r['ontology_region_agreement'] for r in rows if r['ontology_region_agreement'] is not None) if any(r['ontology_region_total_tokens'] for r in rows) else None,
                        'mean_page_anchor_agreement': statistics.mean(r['ontology_anchor_agreement'] for r in rows if r['ontology_anchor_agreement'] is not None) if any(r['ontology_region_total_tokens'] for r in rows) else None,
                        'pages_with_field_values': sum(r['ontology_field_total'] > 0 for r in rows),
                        'mean_page_field_value_agreement': statistics.mean(r['ontology_field_value_agreement'] for r in rows if r['ontology_field_value_agreement'] is not None) if any(r['ontology_field_total'] for r in rows) else None,
                        'mean_page_ontology_weighted_overlap': statistics.mean(r['ontology_weighted_overlap'] for r in rows),
                        'mean_page_token_overlap': statistics.mean(r['token_overlap'] for r in rows),
                        'mean_page_sequence_similarity': statistics.mean(r['sequence_similarity'] for r in rows),
                        'mean_page_combined_similarity': statistics.mean(r['combined_similarity'] for r in rows)})
    return {'schema_version': 6, 'algorithm_version': ALGORITHM_VERSION, 'meaning': 'Text consensus, not OCR accuracy',
            'settings': {'ontology': ontology_metadata, 'entity_weight': entity_weight,
                         'ontology_mode': ontology_mode,
                         'region_policy': region_policy(region_context_tokens) if ontology_mode == 'regions' and ontology_path is not None else None,
                         'primary_metric': primary_metric(ontology_path, ontology_mode, overlap_weight),
                         'overlap_weight': overlap_weight, 'sequence_weight': 1 - overlap_weight,
                         'document_weighting': weighting, 'empty_pair_policy': 'zero when other methods have text'},
            'models': models, 'documents': list(normalized), 'ranking': rank(overall),
            'per_document_rankings': {d: rank(v) for d, v in doc_scores.items()},
            'localization': localizations, 'excluded_pages': excluded, 'pairwise_summary': summary}, comparisons, volumes, normalized
