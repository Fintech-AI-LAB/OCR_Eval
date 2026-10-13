"""Portable full-text evaluation of saved OCR, without ontology or inference."""
from collections import Counter
from copy import deepcopy

from .corroboration import (corroborate_saved, corroboration_policy,
                           DEFAULT_MIN_FAMILIES, DEFAULT_MIN_FRACTION)
from .unified import compare_saved


def evaluate_saved(data, families, panel, *, min_families=DEFAULT_MIN_FAMILIES,
                   min_fraction=DEFAULT_MIN_FRACTION, passage_tokens=40):
    """Combine full-document/page agreement with passage corroboration.

    data maps document IDs to method IDs to records with pages and input_format.
    Document-only methods use pages=None and document_text, and receive only
    whole-document scores. Panel voters must have physical pages in every document.
    Scores are independent of the configurable passage support threshold.
    """
    corroboration_policy(min_families, min_fraction, passage_tokens)
    if not isinstance(data, dict) or not data:
        raise ValueError('At least one document is required.')
    if not isinstance(families, dict) or not families:
        raise ValueError('An engine-family mapping is required.')
    if any(not isinstance(v, (list, tuple)) or not v
           or any(not isinstance(f, str) or not f for f in v) for v in families.values()):
        raise ValueError('Every method needs nonempty engine-family names.')
    if not isinstance(panel, (list, tuple)) or not panel or any(not isinstance(m, str) for m in panel):
        raise ValueError('Panel must be a nonempty list of method names.')
    if len(panel) != len(set(panel)) or not set(panel) <= set(families):
        raise ValueError('Panel must name distinct, available methods.')
    if any(len(families[m]) != 1 for m in panel):
        raise ValueError('Hybrid workflows cannot serve as independent panel voters.')
    if len({families[m][0] for m in panel}) < 2:
        raise ValueError('At least two distinct panel engine families are required.')
    prepared = deepcopy(data)
    aligned_methods = None
    for records in prepared.values():
        if not isinstance(records, dict) or set(records) != set(families):
            raise ValueError('Every document must contain the same methods as families.')
        aligned = set()
        for method, record in records.items():
            if not isinstance(record, dict) or 'pages' not in record:
                raise ValueError('Every method record must supply pages (or None for document-only text).')
            record.setdefault('input_format', 'markdown')
            if record['pages'] is None:
                if not isinstance(record.get('document_text'), str):
                    raise ValueError('Document-only methods must supply document_text.')
            else:
                if not isinstance(record['pages'], list) or not record['pages']:
                    raise ValueError('Aligned methods must supply a nonempty page list.')
                aligned.add(method)
        if aligned_methods is not None and aligned != aligned_methods:
            raise ValueError('Page alignment availability must be consistent across documents.')
        aligned_methods = aligned
    if not set(panel) <= aligned_methods:
        raise ValueError('Panel voters must have physical page alignment.')
    aligned_data = {d: {m: r for m, r in records.items() if m in aligned_methods}
                    for d, records in prepared.items()}
    aligned_families = {m: f for m, f in families.items() if m in aligned_methods}
    corroboration = corroborate_saved(aligned_data, aligned_families, panel,
        min_families=min_families, min_fraction=min_fraction, passage_tokens=passage_tokens)
    document = compare_saved(prepared, families, panel, whole_document=True)
    page = compare_saved(aligned_data, aligned_families, panel)
    counts = {m: Counter() for m in aligned_methods}
    for row in corroboration['passages']:
        counts[row['method']][row['status']] += 1
        counts[row['method']]['flagged'] += bool(row['flags'])
    ranking = []
    for method, score in document['scores'].items():
        count = counts.get(method)
        ranking.append({'method': method, 'score': score,
            'page_agreement': page['scores'].get(method),
            'corroborated_passages': count['corroborated'] if count is not None else None,
            'uncertain_passages': count['uncertain'] if count is not None else None,
            'flagged_passages': count['flagged'] if count is not None else None})
    ranking.sort(key=lambda r: (r['score'] is None, -(r['score'] or 0), r['method']))
    return {'meaning': 'Full-text peer agreement without ontology; corroboration is a separate review diagnostic.',
            'ranking': ranking, 'document_agreement': document, 'page_agreement': page,
            'passage_corroboration': corroboration}
