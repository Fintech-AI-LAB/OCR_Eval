"""Score agreement among OCR output files for one physical page (0–100)."""
import itertools
import json
import math
import statistics
from pathlib import Path

from .ontology import DEFAULT_ONTOLOGY, DEFAULT_ENTITY_WEIGHT, ontology_metrics, load_ontology, validate_entity_weight
from .field_values import extract_field_values, field_text, field_value_metrics, primary_metric, primary_overlap, validate_ontology_mode
from .regions import extract_ontology_regions, region_metrics, region_policy, validate_region_context, DEFAULT_REGION_CONTEXT_TOKENS
from .text import ALGORITHM_VERSION, normalize_texts, compare_page, tokens


def _json_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ('pages', 'full_page_text'):
            if key in value:
                pages = value[key]
                if not isinstance(pages, list) or len(pages) != 1:
                    raise ValueError('JSON must contain exactly one page; split multi-page outputs first.')
                return _json_text(pages[0])
        if isinstance(value.get('markdown'), str):
            text = value['markdown']
            for table in value.get('tables') or []:
                text = text.replace(f'[{table["id"]}]({table["id"]})', table['content'])
            return text
        if isinstance(value.get('text'), str):
            return value['text']
        if isinstance(value.get('lines'), list):
            return '\n'.join(_json_text(line) for line in value['lines'])
    # MinerU's single-page list of content blocks.
    if isinstance(value, list) and all(isinstance(b, dict) and
            isinstance(b.get('content'), (str, type(None))) and 'content' in b for b in value):
        return '\n'.join(b['content'] or '' for b in value)
    raise ValueError('Unsupported page JSON; provide a text/Markdown file or a supported single-page OCR JSON.')


def _json_format(value):
    if isinstance(value, dict):
        for key in ('pages', 'full_page_text'):
            if key in value and isinstance(value[key], list) and len(value[key]) == 1:
                return _json_format(value[key][0])
        if 'markdown' in value:
            return 'markdown'
        if any(key in value for key in ('full_page_text', 'text', 'lines')):
            return 'plain'
    return 'markdown'


def _read_page(path):
    path = Path(path)
    if path.suffix.lower() not in ('.txt', '.md', '.markdown', '.html', '.htm', '.json'):
        raise ValueError(f'{path}: expected an OCR text, Markdown, HTML or JSON file.')
    text = path.read_text(encoding='utf-8-sig')
    suffix = path.suffix.lower()
    if suffix == '.json':
        value = json.loads(text)
        return _json_text(value), _json_format(value)
    return text, 'plain' if suffix == '.txt' else 'html' if suffix in ('.html', '.htm') else 'markdown'


def _evaluate_page(file_paths, overlap_weight=1.0, diagnostics=True,
                   ontology_path=DEFAULT_ONTOLOGY, entity_weight=DEFAULT_ENTITY_WEIGHT, ontology_mode='regions',
               region_context_tokens=DEFAULT_REGION_CONTEXT_TOKENS):
    """Return the primary score, per-file scores, and separate pair diagnostics.

    Inputs must be distinct OCR output files for the same physical page.
    At least two outputs are required; with two, their scores are identical.
    Raises ValueError when all outputs have no scorable word/number tokens.
    Empty pairs receive zero if another output contains scorable text.
    """
    if not isinstance(file_paths, (list, tuple)) or len(file_paths) < 2:
        raise ValueError('Provide an array of at least two file paths for one page.')
    if not math.isfinite(overlap_weight) or not 0 <= overlap_weight <= 1:
        raise ValueError('overlap_weight must be finite and between 0 and 1.')
    validate_entity_weight(entity_weight)
    validate_ontology_mode(ontology_mode)
    validate_region_context(region_context_tokens)
    _, ontology_metadata = load_ontology(ontology_path)
    paths = [Path(p).resolve() for p in file_paths]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate file paths would inflate agreement.')
    sources = [_read_page(p) for p in paths]
    structured = [field_text(s[0], s[1]) for s in sources]
    fields = [extract_field_values(t, ontology_path) if ontology_mode == 'values' else {'fields': [], 'issues': []} for t in structured]
    regions = [extract_ontology_regions(t, ontology_path if ontology_mode == 'regions' else None, context_tokens=region_context_tokens) for t in structured]
    texts = normalize_texts(structured, ['plain'] * len(sources))
    has_tokens = [bool(tokens(text)) for text in texts]
    if not any(has_tokens):
        raise ValueError('All outputs lack scorable text; exclude this page from the average.')
    if ontology_path is not None and ontology_mode == 'values' and not any(f['fields'] or f['issues'] for f in fields):
        raise ValueError('No ontology field values extracted; provide labelled values or use ontology_mode=mentions.')
    if ontology_path is not None and ontology_mode == 'regions' and not any(r['regions'] for r in regions):
        raise ValueError('No ontology regions located; check ontology labels or evaluate without an ontology.')
    scores = [[] for _ in paths]
    pairs = []
    for a, b in itertools.combinations(range(len(paths)), 2):
        if not diagnostics and overlap_weight == 1:
            if ontology_path is not None and ontology_mode == 'regions':
                similarity = region_metrics(regions[a], regions[b])['ontology_region_agreement'] or 0.0
            elif ontology_path is not None and ontology_mode == 'values':
                similarity = field_value_metrics(fields[a], fields[b])['ontology_field_value_agreement'] or 0.0
            else:
                similarity = ontology_metrics(tokens(texts[a]), tokens(texts[b]), ontology_path, entity_weight)['ontology_weighted_overlap']
        else:
            pair = compare_page('page', 1, str(paths[a]), str(paths[b]), texts[a], texts[b], ontology_path, entity_weight,
                                left_fields=fields[a], right_fields=fields[b], left_regions=regions[a], right_regions=regions[b],
                                ontology_mode=ontology_mode, region_context_tokens=region_context_tokens)
            similarity = (overlap_weight * primary_overlap(pair, ontology_path, ontology_mode) +
                          (1 - overlap_weight) * pair['sequence_similarity'])
            if diagnostics:
                pair['primary_similarity'] = similarity
                pairs.append(pair)
        scores[a].append(100 * similarity)
        scores[b].append(100 * similarity)
    model_scores = [statistics.mean(values) for values in scores]
    return {'algorithm_version': ALGORITHM_VERSION, 'score': statistics.mean(model_scores),
            'model_scores': model_scores, 'files': [str(p) for p in paths],
            'settings': {'ontology': ontology_metadata, 'entity_weight': entity_weight,
                         'overlap_weight': overlap_weight, 'sequence_weight': 1 - overlap_weight,
                         'ontology_mode': ontology_mode,
                         'region_policy': region_policy(region_context_tokens) if ontology_mode == 'regions' and ontology_path is not None else None,
                         'primary_metric': primary_metric(ontology_path, ontology_mode, overlap_weight)},
            'localization': {str(path): region['localization'] for path, region in zip(paths, regions)} if ontology_mode == 'regions' and ontology_path is not None else {},
            'pairs': pairs}


def score_page_details(file_paths, overlap_weight=1.0, *, ontology_path=DEFAULT_ONTOLOGY,
               entity_weight=DEFAULT_ENTITY_WEIGHT, ontology_mode='regions',
               region_context_tokens=DEFAULT_REGION_CONTEXT_TOKENS):
    """Return scores and pairwise token, sequence, numeric and text-difference diagnostics."""
    return _evaluate_page(file_paths, overlap_weight, diagnostics=True, ontology_path=ontology_path, entity_weight=entity_weight, ontology_mode=ontology_mode, region_context_tokens=region_context_tokens)


def score_models(file_paths, overlap_weight=1.0, *, ontology_path=DEFAULT_ONTOLOGY,
               entity_weight=DEFAULT_ENTITY_WEIGHT, ontology_mode='regions',
               region_context_tokens=DEFAULT_REGION_CONTEXT_TOKENS):
    """Return one 0–100 primary consensus score per file, in input order."""
    return _evaluate_page(file_paths, overlap_weight, diagnostics=False, ontology_path=ontology_path, entity_weight=entity_weight, ontology_mode=ontology_mode, region_context_tokens=region_context_tokens)['model_scores']


def score_page(file_paths, overlap_weight=1.0, *, ontology_path=DEFAULT_ONTOLOGY,
               entity_weight=DEFAULT_ENTITY_WEIGHT, ontology_mode='regions',
               region_context_tokens=DEFAULT_REGION_CONTEXT_TOKENS):
    """Return a single 0–100 score: mean agreement across all output pairs.

    Supply ontology_path to compare text regions around ontology labels.
    Use ontology_mode="values" for exact field-value comparison.
    Use ontology_mode="mentions" for the legacy label-weighted token score.
    This measures consensus, not accuracy. Average scores to weight pages equally.
    Set overlap_weight explicitly below 1 to add global sequence penalties.
    """
    return statistics.mean(score_models(file_paths, overlap_weight, ontology_path=ontology_path, entity_weight=entity_weight, ontology_mode=ontology_mode, region_context_tokens=region_context_tokens))
