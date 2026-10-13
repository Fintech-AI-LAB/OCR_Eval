"""Conservative, page-aligned passage corroboration by a fixed family panel.

Support means exact normalized tokens at a monotonically aligned location, not
verified image text. No fuzzy match or majority repair changes the source text.
"""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import math

from .field_values import field_text
from .text import clean, tokens

VERSION = '2.1'
DEFAULT_MIN_FAMILIES = 2
DEFAULT_MIN_FRACTION = 0.0


def corroboration_policy(min_families=DEFAULT_MIN_FAMILIES,
                         min_fraction=DEFAULT_MIN_FRACTION, passage_tokens=40):
    if isinstance(min_families, bool) or not isinstance(min_families, int) or min_families < 2:
        raise ValueError('min_families must be an integer of at least 2.')
    if (isinstance(min_fraction, bool) or not isinstance(min_fraction, (int, float))
            or not math.isfinite(min_fraction) or not 0 <= min_fraction <= 1):
        raise ValueError('min_fraction must be finite and in [0, 1]; 0 disables the fraction requirement.')
    if isinstance(passage_tokens, bool) or not isinstance(passage_tokens, int) or not 1 <= passage_tokens <= 128:
        raise ValueError('passage_tokens must be an integer from 1 to 128.')
    return {
        'version': VERSION, 'min_families': min_families,
        'min_fraction': min_fraction, 'passage_tokens': passage_tokens,
        'threshold': 'max(min_families, ceil(min_fraction * all_panel_families))',
        'family_vote': 'All configured panel members of a family must complete the page and support the passage.',
        'source_vote': 'The source counts only if it belongs to the fixed panel; its family still has at most one vote.',
        'alignment': 'Monotonic exact token alignment within a physical page; repeated matches require unique context (up to 8 tokens per side).',
        'normalization': 'Formatting, case and ordinary punctuation ignored; signed numeric tokens preserved. No peer-based repairs.',
        'passages': 'Decoded source lines, split at the token limit; offsets are zero-based, end-exclusive normalized page tokens.',
        'uncertainty': 'Below the family support threshold only. Conflicts, ambiguous alignment and incomplete source output are separate flags, never vetoes.',
        'meaning': 'Uncalibrated corroboration heuristic, not accuracy, text detection or proof of independent errors.',
    }


def _prepare(text, input_format, limit):
    words, spans = [], []
    for line in field_text(text, input_format).splitlines():
        line_words = tokens(clean(line))
        offset = len(words)
        words.extend(line_words)
        spans.extend((start, min(start + limit, len(words)))
                     for start in range(offset, len(words), limit))
    return words, spans


def _unique(words, phrase, cache):
    key = tuple(phrase)
    if key not in cache:
        count = 0
        for i in range(len(words) - len(phrase) + 1):
            if words[i:i + len(phrase)] == phrase:
                count += 1
                if count == 2:
                    break
        cache[key] = count == 1
    return cache[key]


def _evidence(source, peer, start, end, opcodes, source_cache, peer_cache):
    for tag, a, b, c, d in opcodes:
        if tag == 'equal' and a <= start and end <= b:
            left, right = c + start - a, c + end - a
            context = source[max(a, start - 8):min(b, end + 8)]
            unique = (_unique(source, context, source_cache)
                      and _unique(peer, context, peer_cache))
            return {'state': 'support' if unique else 'ambiguous',
                    'peer_token_start': left, 'peer_token_end': right,
                    'peer_text': ' '.join(peer[left:right])}
    relevant = [(tag, a, b, c, d) for tag, a, b, c, d in opcodes
                if max(start, a) < min(end, b) or (tag == 'insert' and start < a < end)]
    conflict = any(tag in ('replace', 'insert') for tag, *_ in relevant)
    # Clip equal blocks to this passage; replacements retain their full span so
    # reviewers can inspect alternate readings without pretending exact offsets.
    intervals = [(c + max(0, start - a), d - max(0, b - end)) if tag == 'equal' else (c, d)
                 for tag, a, b, c, d in relevant]
    left = min((a for a, _ in intervals), default=0)
    right = max((b for _, b in intervals), default=left)
    return {'state': 'conflict' if conflict else 'missing',
            'peer_token_start': left, 'peer_token_end': right,
            'peer_text': ' '.join(peer[left:min(right, left + 80)]),
            'peer_text_truncated': right - left > 80}


def _family_state(evidence):
    states = {e['state'] for e in evidence}
    if states == {'support'}:
        return 'support'
    if 'conflict' in states or {'support', 'missing'} <= states:
        return 'conflict'
    if 'ambiguous' in states:
        return 'ambiguous'
    if 'unavailable' in states:
        return 'unavailable'
    return 'missing'


def corroborate_saved(data, families, panel, *, min_families=DEFAULT_MIN_FAMILIES,
                      min_fraction=DEFAULT_MIN_FRACTION, passage_tokens=40):
    """Return every source passage, its family evidence and uncertainty reasons.

    Input has the same document/method/page structure as ``compare_saved``.
    Only panel methods vote. Hybrid methods may be evaluated, but cannot vote.
    All configured families remain in the threshold denominator even on failure.
    By default, two families suffice regardless of panel size. A nonzero
    min_fraction adds an optional proportional floor.
    Missing statuses default to complete, for callers with plain saved text.
    Unaligned document-only methods must be removed by the caller.
    """
    policy = corroboration_policy(min_families, min_fraction, passage_tokens)
    if not panel or len(panel) != len(set(panel)) or not set(panel) <= set(families):
        raise ValueError('Panel must name distinct, available methods.')
    if any(not isinstance(v, (list, tuple)) or not v
           or any(not isinstance(f, str) or not f for f in v) for v in families.values()):
        raise ValueError('Every method needs nonempty engine-family names.')
    members = defaultdict(list)
    for method in sorted(panel):
        if len(families[method]) != 1:
            raise ValueError('Hybrid workflows cannot serve as independent panel voters.')
        members[families[method][0]].append(method)
    family_names = sorted(members)
    required = max(min_families, math.ceil(min_fraction * len(members)))
    rows, pages = [], []
    for document, records in sorted(data.items()):
        if set(records) != set(families):
            raise ValueError('Every document must contain the same methods.')
        if any(not isinstance(r.get('pages'), list) for r in records.values()):
            raise ValueError('Physical page alignment required.')
        counts = {len(r['pages']) for r in records.values()}
        if len(counts) != 1:
            raise ValueError('Physical page counts must match.')
        count = next(iter(counts))
        for record in records.values():
            if not all(isinstance(t, str) for t in record['pages']):
                raise ValueError('Page text must be a string.')
            if 'statuses' in record and (len(record['statuses']) != count
                    or any(s not in ('complete', 'failed', 'truncated') for s in record['statuses'])):
                raise ValueError('Page statuses must be aligned complete/failed/truncated values.')
        for index in range(count):
            prepared = {m: _prepare(r['pages'][index], r.get('input_format', 'markdown'), passage_tokens)
                        for m, r in records.items()}
            states = {m: r['statuses'][index] if 'statuses' in r else 'complete'
                      for m, r in records.items()}
            alignments, caches = {}, {m: {} for m in records}

            def align(a, b):
                key = tuple(sorted((a, b)))
                if key not in alignments:
                    alignments[key] = SequenceMatcher(None, prepared[key[0]][0], prepared[key[1]][0],
                                                      autojunk=False).get_opcodes()
                codes = alignments[key]
                if a == key[0]:
                    return codes
                return [({'insert': 'delete', 'delete': 'insert'}.get(t, t), c, d, x, y)
                        for t, x, y, c, d in codes]

            for method in sorted(records):
                source, spans = prepared[method]
                method_rows = []
                for passage_index, (start, end) in enumerate(spans, 1):
                    evidence = {}
                    for peer in sorted(panel):
                        if states[peer] != 'complete':
                            evidence[peer] = {'state': 'unavailable', 'page_status': states[peer]}
                        elif peer == method:
                            evidence[peer] = {'state': 'support', 'peer_token_start': start,
                                              'peer_token_end': end, 'peer_text': ' '.join(source[start:end])}
                        else:
                            evidence[peer] = _evidence(source, prepared[peer][0], start, end,
                                                      align(method, peer), caches[method], caches[peer])
                    family_evidence = {f: {'state': _family_state([evidence[m] for m in members[f]]),
                                           'members': {m: evidence[m] for m in members[f]}}
                                       for f in family_names}
                    groups = {state: [f for f in family_names if family_evidence[f]['state'] == state]
                              for state in ('support', 'conflict', 'ambiguous', 'missing', 'unavailable')}
                    reasons = []
                    flags = []
                    if len(groups['support']) < required:
                        reasons.append('insufficient_family_support')
                    if groups['conflict']:
                        flags.append('conflicting_readings')
                    if groups['ambiguous']:
                        flags.append('ambiguous_alignment')
                    if states[method] != 'complete':
                        flags.append('incomplete_source')
                    row = {'document': document, 'page': index + 1, 'method': method,
                           'passage': passage_index, 'token_start': start, 'token_end': end,
                           'text': ' '.join(source[start:end]),
                           'context_before': ' '.join(source[max(0, start - 8):start]),
                           'context_after': ' '.join(source[end:end + 8]),
                           'source_status': states[method],
                           'supporting_family_count': len(groups['support']),
                           'total_family_count': len(members), 'required_family_count': required,
                           'other_family_support_count': len(set(groups['support']) - set(families[method])),
                           'supporting_families': groups['support'], 'conflicting_families': groups['conflict'],
                           'ambiguous_families': groups['ambiguous'], 'missing_families': groups['missing'],
                           'unavailable_families': groups['unavailable'],
                           'status': 'uncertain' if reasons else 'corroborated',
                           'uncertainty_reasons': reasons, 'flags': flags, 'family_evidence': family_evidence}
                    method_rows.append(row)
                rows.extend(method_rows)
                uncertain = sum(r['status'] == 'uncertain' for r in method_rows)
                pages.append({'document': document, 'page': index + 1, 'method': method,
                              'source_status': states[method], 'passages': len(method_rows),
                              'uncertain_passages': uncertain,
                              'flagged_passages': sum(bool(r['flags']) for r in method_rows),
                              'status': 'uncertain' if uncertain or not spans else 'corroborated',
                              'empty_output': not spans,
                              'uncertainty_reasons': ['no_text_returned'] if not spans else
                                  sorted({r for row in method_rows for r in row['uncertainty_reasons']}),
                              'flags': (['incomplete_source'] if states[method] != 'complete' else []) if not spans else
                                  sorted({f for row in method_rows for f in row['flags']})})
    summary = dict(Counter(r['status'] for r in rows))
    summary = {s: summary.get(s, 0) for s in ('corroborated', 'uncertain')}
    summary['empty_page_outputs'] = sum(p['empty_output'] for p in pages)
    summary['flagged_passages'] = sum(bool(r['flags']) for r in rows)
    summary['corroborated_with_flags'] = sum(r['status'] == 'corroborated' and bool(r['flags']) for r in rows)
    return {'policy': policy, 'panel': sorted(panel), 'family_members': dict(sorted(members.items())),
            'required_family_count': required, 'summary': summary, 'pages': pages, 'passages': rows}


def passage_csv_rows(result):
    """Flatten the review fields; detailed peer spans/readings stay in JSON."""
    for row in result['passages']:
        yield {key: '; '.join(value) if isinstance(value, list) else value
               for key, value in row.items() if key != 'family_evidence'}
