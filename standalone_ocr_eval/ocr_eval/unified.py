"""Fixed-panel, order-sensitive agreement diagnostics for saved OCR workflows.

No SequenceMatcher, reference text, or cross-model normalization is used. A new
candidate never becomes a voter unless the caller explicitly revises the panel.
"""
from collections import Counter, defaultdict
from itertools import combinations
import statistics

from .text import clean, tokens, counter_overlap
from .field_values import field_text

VERSION = '1.0'


def signature(text, input_format='markdown'):
    words = tokens(clean(field_text(text, input_format)))
    return Counter(words), Counter(zip(words, words[1:]))


def agreement(a, b):
    unigram = counter_overlap(a[0], b[0])
    # A one-token identical pair remains perfect; an empty pair is always zero.
    bigram = counter_overlap(a[1], b[1]) if a[1] or b[1] else unigram
    return {'token_dice': unigram, 'bigram_dice': bigram,
            'similarity': (unigram + bigram) / 2}


def fixed_panel_scores(pair_scores, families, panel):
    """Equal other-family votes, excluding any shared ancestry (including hybrids)."""
    if len(panel) != len(set(panel)) or not set(panel) <= set(families):
        raise ValueError('Panel must name distinct, available methods.')
    scores = {}
    for method, own in families.items():
        peers = defaultdict(list)
        for peer in panel:
            if peer == method or set(own) & set(families[peer]):
                continue
            if len(families[peer]) != 1:
                raise ValueError('Hybrid workflows cannot serve as independent panel voters.')
            peers[families[peer][0]].append(pair_scores[tuple(sorted((method, peer)))])
        scores[method] = 100 * statistics.mean(statistics.mean(v) for v in peers.values()) if peers else None
    return scores


def compare_saved(data, families, panel, *, page_mask=None, whole_document=False):
    """Average selected physical pages within documents, then documents equally.

    data: document -> method -> {pages, input_format}; unaligned methods supply
    document_text and pages=None and may only enter the whole-document track.
    Empty/failed pages remain in the full-corpus denominator.
    """
    methods = sorted(families)
    by_pair = defaultdict(list)
    rows, per_document = [], {}
    for document, records in data.items():
        if set(records) != set(methods):
            raise ValueError('Every document must contain the same methods.')
        if whole_document:
            units = [0]
        else:
            counts = {len(r['pages']) for r in records.values() if r['pages'] is not None}
            if any(r['pages'] is None for r in records.values()) or len(counts) != 1:
                raise ValueError('Physical page alignment required.')
            units = list(range(next(iter(counts)))) if page_mask is None else page_mask[document]
        if not units:
            continue
        signatures = {}
        for method, record in records.items():
            texts = ([record.get('document_text') if record['pages'] is None else '\n\n'.join(record['pages'])]
                     if whole_document else [record['pages'][i] for i in units])
            signatures[method] = [signature(t, record['input_format']) for t in texts]
        document_pairs = {}
        for a, b in combinations(methods, 2):
            if a not in panel and b not in panel:
                continue  # Candidate-only edges cannot affect fixed-panel scores.
            scores = []
            for i, unit in enumerate(units):
                result = agreement(signatures[a][i], signatures[b][i])
                scores.append(result['similarity'])
                rows.append({'document': document, 'page': None if whole_document else unit + 1,
                             'left': a, 'right': b, **result})
            document_pairs[a, b] = statistics.mean(scores)
            by_pair[a, b].append(document_pairs[a, b])
        per_document[document] = fixed_panel_scores(document_pairs, families, panel)
    if not by_pair:
        raise ValueError('No common evaluation units.')
    pairs = {key: statistics.mean(values) for key, values in by_pair.items()}
    scores = fixed_panel_scores(pairs, families, panel)
    sensitivity = {family: fixed_panel_scores(pairs, families, [p for p in panel if family not in families[p]])
                   for family in sorted({f for p in panel for f in families[p]})}
    return {'version': VERSION, 'meaning': 'Fixed-panel family-balanced agreement; not accuracy.',
            'metric': '0.5 token Dice + 0.5 ordered bigram Dice',
            'normalization': 'Independent per output; no peer-corroborated repairs.',
            'panel': panel, 'families': families, 'scores': scores,
            'per_document': per_document, 'leave_one_panel_family_out': sensitivity,
            'unit': 'whole_document' if whole_document else 'physical_page',
            'pair_policy': 'Only pairs touching the fixed panel; no candidate-only comparisons.',
            'document_weighting': 'equal', 'pairs': rows}
