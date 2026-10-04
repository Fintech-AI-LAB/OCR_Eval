"""Ontology-anchored text regions, with no exact field/value extraction."""
from collections import Counter
import math
import re

if __package__:
    from .ontology_metrics import DEFAULT_ONTOLOGY
    from .field_values import _load_schema, _matches, field_text
else:
    from ontology_metrics import DEFAULT_ONTOLOGY
    from field_values import _load_schema, _matches, field_text

DEFAULT_REGION_CONTEXT_TOKENS = 16
MIN_CONTEXT_WORD_DICE = 0.5
STOP_LABELS = set('a an the to from of in on at by as for and or is are be with without this that it its not no yes'.split())
NUMBER = r'(?:\(\s*[+\-−]?\d+(?:[.,:/-]\d+)*(?:[%‰])?\s*\)|[+\-−]?\d+(?:[.,:/-]\d+)*(?:[%‰])?)'
TOKEN = re.compile(r'(?<!\w)' + NUMBER + r'(?!\w)|\w+')


def validate_region_context(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 128:
        raise ValueError('region_context_tokens must be an integer from 1 to 128.')


def region_policy(context_tokens=DEFAULT_REGION_CONTEXT_TOKENS):
    return {'context_tokens_each_side': context_tokens,
            'boundaries': 'one local window per canonical anchor; no fixed-size chunks',
            'overlap_accounting': 'each token and bigram divided equally among covering windows',
            'alignment': 'maximum-weight one-to-one matching within the same central anchor tag',
            'ambiguous_anchor_gate': {'exact_window_or_shared_unambiguous_neighbor': True,
                                      'otherwise_min_context_word_dice': MIN_CONTEXT_WORD_DICE,
                                      'min_distinct_shared_context_words': 2},
            'similarity': '0.5 overlap-weighted token Dice + 0.5 overlap-weighted ordered bigram Dice',
            'aggregation': 'region-token-mass weighted; unmatched regions receive zero',
            'localization': 'lexical ontology anchors; common function-word labels suppressed',
            'meaning': 'text-region consensus; no field-value parsing or spatial coordinates'}


def _tag(candidates):
    roots = sorted({c['field'] for c in candidates if c['owner'] is None})
    if roots:
        return 'concept:' + '|'.join(roots).casefold()
    names = sorted({re.sub(r'[^\w]', '', c['name'].casefold()).replace('_', '') for c in candidates})
    return 'label:' + '|'.join(names)


def extract_ontology_regions(text, ontology_path=DEFAULT_ONTOLOGY, input_format='plain',
                             context_tokens=DEFAULT_REGION_CONTEXT_TOKENS):
    """Select local text windows around ontology mentions, independently per output.

    Canonical anchors normalize declared aliases/CamelCase forms. Ambiguous owners
    stay in evidence; selecting a passage does not assert an exact field identity.
    """
    validate_region_context(context_tokens)
    trie = _load_schema(ontology_path)
    source = field_text(text, input_format)
    matches = []
    suppressed = []
    for start, end, candidates in _matches(source, trie):
        label = source[start:end]
        compact = re.sub(r'\s+', ' ', label.casefold()).strip()
        if compact in STOP_LABELS or len(re.sub(r'\W', '', compact)) < 3:
            suppressed.append({'label': label, 'start': start, 'end': end})
            continue
        matches.append({'label': label, 'start': start, 'end': end, 'tag': _tag(candidates),
                        'candidate_fields': sorted({c['field'] for c in candidates}),
                        'ambiguous_owner': len({c['field'] for c in candidates}) > 1})
    # One token per canonical label, retaining every surrounding numeric/text token.
    tokens, spans, anchors = [], [], []
    cursor = 0
    for match in matches:
        for token in TOKEN.finditer(source, cursor, match['start']):
            value = token[0].casefold().replace('−', '-')
            tokens.append(re.sub(r'\s+', '', value) if value.startswith('(') else value)
            spans.append((token.start(), token.end()))
        match['token_offset'] = len(tokens)
        tokens.append('@' + match['tag'])
        spans.append((match['start'], match['end']))
        anchors.append(match)
        cursor = match['end']
    for token in TOKEN.finditer(source, cursor):
        value = token[0].casefold().replace('−', '-')
        tokens.append(re.sub(r'\s+', '', value) if value.startswith('(') else value)
        spans.append((token.start(), token.end()))
    # Anchor-relative windows avoid cascading boundary shifts after an insertion.
    # They overlap for contextual comparison, but their token/bigram budgets do
    # not: every selected occurrence has total weight one across all windows.
    intervals = [(max(0, a['token_offset'] - context_tokens),
                  min(len(tokens), a['token_offset'] + context_tokens + 1)) for a in anchors]
    token_coverage, bigram_coverage = [0] * len(tokens), [0] * max(0, len(tokens) - 1)
    for begin, end in intervals:
        for pos in range(begin, end):
            token_coverage[pos] += 1
        for pos in range(begin, end - 1):
            bigram_coverage[pos] += 1
    regions = []
    for anchor, (begin, end) in zip(anchors, intervals):
        selected = [a for a in anchors if begin <= a['token_offset'] < end]
        start_char, end_char = spans[begin][0], spans[end - 1][1]
        token_weights = [1 / token_coverage[pos] for pos in range(begin, end)]
        regions.append({'region': len(regions) + 1, 'start_token': begin, 'end_token': end,
                        'start_char': start_char, 'end_char': end_char,
                        'start_line': source.count('\n', 0, start_char) + 1,
                        'end_line': source.count('\n', 0, end_char) + 1,
                        'text': source[start_char:end_char], 'tokens': tokens[begin:end],
                        'token_weights': token_weights,
                        'bigram_weights': [1 / bigram_coverage[pos] for pos in range(begin, end - 1)],
                        'token_mass': math.fsum(token_weights), 'central_anchor': anchor,
                        'anchor_tags': sorted({a['tag'] for a in selected}), 'anchors': selected})
    selected_count = sum(count > 0 for count in token_coverage)
    return {'regions': regions, 'anchors': anchors, 'suppressed_anchors': suppressed,
            'localization': {'anchor_count': len(anchors), 'region_count': len(regions),
                             'total_tokens': len(tokens), 'selected_tokens': selected_count,
                             'selected_token_fraction': selected_count / len(tokens) if tokens else None,
                             'ambiguous_anchor_count': sum(a['ambiguous_owner'] for a in anchors)}}


def _dice(a, b):
    total = sum(a.values()) + sum(b.values())
    return min(1.0, 2 * math.fsum((a & b).values()) / total) if total else None


def _assignment(weights):
    """Maximum-weight rectangular assignment with zero-valued dummy matches."""
    rows = len(weights)
    columns = len(weights[0]) if rows else 0
    size = max(rows, columns)
    if not size:
        return []
    u, v, p, way = [0.0]*(size+1), [0.0]*(size+1), [0]*(size+1), [0]*(size+1)
    for i in range(1, size+1):
        p[0], j0 = i, 0
        minimum, used = [float('inf')]*(size+1), [False]*(size+1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float('inf'), 0
            for j in range(1, size+1):
                if used[j]:
                    continue
                weight = weights[i0-1][j-1] if i0 <= rows and j <= columns else 0.0
                current = -weight - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j], way[j] = current, j0
                if minimum[j] < delta:
                    delta, j1 = minimum[j], j
            for j in range(size+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j]-1, j-1) for j in range(1, size+1) if 0 < p[j] <= rows and j <= columns]


def _weighted_counters(region):
    words, bigrams = Counter(), Counter()
    for word, weight in zip(region['tokens'], region['token_weights']):
        words[word] += weight
    for pair, weight in zip(zip(region['tokens'], region['tokens'][1:]), region['bigram_weights']):
        bigrams[pair] += weight
    return words, bigrams


def _context(region):
    central_tag = region['central_anchor']['tag']
    neighbors = {a['tag'] for a in region['anchors']
                 if a['tag'] != central_tag and not a['ambiguous_owner']}
    words = Counter(t for t in region['tokens'] if not t.startswith('@')
                    and t not in STOP_LABELS and len(t) >= 3 and any(c.isalpha() for c in t))
    return neighbors, words


def _context_evidence(x, y, xc, yc):
    """A generic shared label alone cannot connect unrelated passages."""
    required = x['central_anchor']['ambiguous_owner'] or y['central_anchor']['ambiguous_owner']
    neighbors = sorted(xc[0] & yc[0])
    words = sorted(set(xc[1]) & set(yc[1]))
    word_dice = _dice(xc[1], yc[1]) or 0.0
    exact = x['tokens'] == y['tokens']
    eligible = (not required or exact or bool(neighbors)
                or (len(words) >= 2 and word_dice >= MIN_CONTEXT_WORD_DICE))
    return {'required': required, 'eligible': eligible, 'exact_window': exact,
            'shared_unambiguous_neighbor_tags': neighbors,
            'shared_context_words': words, 'context_word_overlap': word_dice}


def region_metrics(left, right):
    """Match anchor-relative passages with a nonduplicated selected-text budget."""
    a, b = left['regions'], right['regions']
    ac, bc = list(map(_weighted_counters, a)), list(map(_weighted_counters, b))
    ax, bx = list(map(_context, a)), list(map(_context, b))
    # Different central labels never compete in the same assignment. Besides
    # enforcing correspondence, grouping avoids a document-sized cubic matrix.
    groups = {}
    for i, x in enumerate(a):
        groups.setdefault(x['central_anchor']['tag'], ([], []))[0].append(i)
    for j, y in enumerate(b):
        groups.setdefault(y['central_anchor']['tag'], ([], []))[1].append(j)
    matched, details, contributions = [], {}, []
    rejected_count, rejected_examples = 0, []
    for tag, (indices_a, indices_b) in sorted(groups.items()):
        if not indices_a or not indices_b:
            continue
        weights = []
        for i in indices_a:
            row = []
            for j in indices_b:
                evidence = _context_evidence(a[i], b[j], ax[i], bx[j])
                if not evidence['eligible']:
                    row.append(0.0)
                    rejected_count += 1
                    if len(rejected_examples) < 20:
                        rejected_examples.append({'left_region': a[i]['region'], 'right_region': b[j]['region'],
                                                  'central_anchor_tag': tag, 'context_evidence': evidence})
                    continue
                token_overlap = _dice(ac[i][0], bc[j][0]) or 0.0
                bigram_overlap = _dice(ac[i][1], bc[j][1])
                similarity = token_overlap if bigram_overlap is None else (token_overlap + bigram_overlap) / 2
                mass = a[i]['token_mass'] + b[j]['token_mass']
                row.append(similarity * mass)
                details[i, j] = {'left_region': a[i]['region'], 'right_region': b[j]['region'],
                                'central_anchor_tag': tag, 'context_evidence': evidence,
                                'shared_anchor_tags': sorted(set(a[i]['anchor_tags']) & set(b[j]['anchor_tags'])),
                                'token_overlap': token_overlap, 'bigram_overlap': bigram_overlap,
                                'similarity': similarity, 'token_mass': mass}
            weights.append(row)
        for row, column in _assignment(weights):
            if weights[row][column] > 0:
                matched.append((indices_a[row], indices_b[column]))
                contributions.append(weights[row][column])
    matched.sort()
    matched_left, matched_right = {i for i, _ in matched}, {j for _, j in matched}
    total = left['localization']['selected_tokens'] + right['localization']['selected_tokens']
    shared_mass = min(total, math.fsum(contributions))
    left_tags, right_tags = Counter(x['tag'] for x in left['anchors']), Counter(x['tag'] for x in right['anchors'])
    return {'ontology_region_agreement': shared_mass / total if total else None,
            'ontology_region_total_tokens': total, 'ontology_region_matched_mass': shared_mass,
            'ontology_anchor_agreement': _dice(left_tags, right_tags),
            'left_region_localization': left['localization'], 'right_region_localization': right['localization'],
            'left_ontology_regions': a, 'right_ontology_regions': b,
            'region_matches': [details[i, j] for i, j in matched],
            'context_rejected_candidate_count': rejected_count,
            'context_rejected_candidate_examples': rejected_examples,
            'unmatched_left_ontology_regions': [r for i, r in enumerate(a) if i not in matched_left],
            'unmatched_right_ontology_regions': [r for j, r in enumerate(b) if j not in matched_right],
            'unmatched_left_anchor_counts': dict(left_tags-right_tags),
            'unmatched_right_anchor_counts': dict(right_tags-left_tags)}
