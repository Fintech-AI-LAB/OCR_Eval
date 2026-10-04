"""Conservative extraction and comparison of ontology-labelled field values.

Only explicit labels, table headers and typed inline values are extracted. No
unlabelled values, fuzzy labels or semantic relationships are inferred.
"""
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unicodedata

if __package__:
    from .ontology_metrics import DEFAULT_ONTOLOGY, load_ontology
else:
    from ontology_metrics import DEFAULT_ONTOLOGY, load_ontology

VALUE_TYPES = {'text', 'identifier', 'number', 'amount', 'date', 'currency'}
WORD = re.compile(r'[^\W_]+', re.UNICODE)
MISSING = {'', '-', '—', 'n/a', 'na', 'null', 'none', '<redacted>', '[redacted]'}
BLOCK_TAGS = set('p div br hr table tr li h1 h2 h3 h4 h5 h6 section article pre blockquote ul ol caption'.split())
FORMAT_TAGS = BLOCK_TAGS | set('td th span strong b em i u a code html body thead tbody tfoot s del sup sub img'.split())


def _clean(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value).replace('\u00ad', '')).strip()


def _variants(name):
    split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    split = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', split)
    return {tuple(WORD.findall(_clean(v).casefold())) for v in (name, split)} - {()}


def _type(name):
    compact = re.sub(r'[^a-z0-9]', '', name.casefold())
    if compact in {'iban', 'bban', 'bic', 'isin', 'lei', 'uetr', 'account'} or any(
            word in compact for word in ('identifier', 'identification', 'accountnumber')):
        return 'identifier'
    if compact.endswith('date'):
        return 'date'
    if 'currency' in compact:
        return 'currency'
    if any(word in compact for word in ('amount', 'balance', 'price', 'quantity', 'rate')):
        return 'amount' if 'amount' in compact or 'balance' in compact else 'number'
    return 'text'


@lru_cache(maxsize=8)
def _schema(path, mtime, size):
    data = json.loads(Path(path).read_bytes())
    trie = {}
    for key, concept in data['concepts'].items():
        aliases = concept.get('aliases', [])
        types = concept.get('field_types', {})
        if not isinstance(aliases, list) or not all(isinstance(x, str) and x.strip() for x in aliases):
            raise ValueError('Ontology aliases must be a list of nonempty strings.')
        if not isinstance(types, dict) or any(not isinstance(v, str) or v not in VALUE_TYPES for v in types.values()):
            raise ValueError('Ontology field_types must map field names to supported value types.')
        if set(types) - set(concept.get('fields', [])):
            raise ValueError('Ontology field_types must refer to declared fields.')
        kind = concept.get('value_type', _type(key))
        order = concept.get('date_order', data.get('date_order'))
        if not isinstance(kind, str) or kind not in VALUE_TYPES or order not in (None, 'dmy', 'mdy'):
            raise ValueError('Invalid ontology value_type or date_order.')
        entries = [(key, None, kind, [key, concept.get('name', key), *aliases])]
        for field in concept.get('fields', []):
            entries.append((f'{key}.{field}', key, types.get(field, _type(field)), [field]))
        # Children which have records use those records' canonical identities.
        for child in concept.get('children', []):
            if child not in data['concepts']:
                entries.append((child, None, _type(child), [child]))
        for identity, owner, value_type, names in entries:
            spec = {'field': identity, 'owner': owner, 'value_type': value_type,
                    'date_order': order, 'name': names[0]}
            for name in names:
                for phrase in _variants(name):
                    node = trie
                    for word in phrase:
                        node = node.setdefault(word, {})
                    candidates = node.setdefault(None, [])
                    if spec not in candidates:
                        candidates.append(spec)
    return trie


def _load_schema(path):
    load_ontology(path)  # Shared base-schema validation and provenance.
    if path is None:
        return {}
    path = Path(path).resolve()
    stat = path.stat()
    return _schema(str(path), stat.st_mtime_ns, stat.st_size)


class _StructuredHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.cells = 0
    def handle_data(self, value):
        self.parts.append(value)
    def handle_starttag(self, tag, attrs):
        if tag in {'td', 'th'}:
            if self.cells:
                self.parts.append('\t')
            self.cells += 1
        elif tag in BLOCK_TAGS:
            self.parts.append('\n')
            if tag == 'tr':
                self.cells = 0
        elif tag not in FORMAT_TAGS:
            self.parts.append(self.get_starttag_text())
    def handle_endtag(self, tag):
        if tag in BLOCK_TAGS:
            self.parts.append('\n')
        elif tag not in FORMAT_TAGS:
            self.parts.append(f'</{tag}>')


def field_text(text, input_format='plain'):
    """Decode formatting once while retaining lines and table cells for binding."""
    if input_format not in {'plain', 'markdown', 'html'}:
        raise ValueError(f'Unsupported input_format: {input_format}')
    if input_format == 'markdown':
        text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
        text = re.sub(r'(?m)^\s{0,3}#{1,6}\s+', '', text)
        text = re.sub(r'(?m)^\s*[-*+]\s+(?=\D)', '', text)
        text = re.sub(r'(?m)^\s*\|?[ :|-]+\|[ :|-]*$', '', text)
        text = text.replace('**', '').replace('`', '')
        text = '\n'.join(line.strip().removeprefix('|').removesuffix('|').replace('|', '\t')
                         if '|' in line else line for line in text.splitlines())
    if input_format != 'plain':
        parser = _StructuredHTML()
        parser.feed(text)
        parser.close()
        text = ''.join(parser.parts)
    text = unicodedata.normalize('NFKC', text).replace('\u00ad', '')
    return '\n'.join(re.sub(r'[^\S\t\n]+', ' ', line).strip(' ') for line in text.splitlines()).strip('\n')


def _matches(text, trie):
    words = list(WORD.finditer(text))
    index = 0
    while index < len(words):
        node, cursor, found = trie, index, None
        while cursor < len(words) and words[cursor][0].casefold() in node:
            # A phrase cannot bridge a colon, equals sign, or sentence boundary.
            if cursor > index and not re.fullmatch(r'[\s_-]*', text[words[cursor-1].end():words[cursor].start()]):
                break
            node = node[words[cursor][0].casefold()]
            cursor += 1
            if None in node:
                found = (cursor, node[None])
        if found:
            end, candidates = found
            yield words[index].start(), words[end-1].end(), candidates
            index = end
        else:
            index += 1


def _resolve(candidates, context):
    scoped = [c for c in candidates if c['owner'] == context] if context else []
    roots = [c for c in candidates if c['owner'] is None]
    selected = scoped or roots or candidates
    identities = {c['field'] for c in selected}
    return selected[0] if len(identities) == 1 else None


def _label(cell, trie, context):
    cell = _clean(cell).rstrip(':=').strip()
    matches = list(_matches(cell, trie))
    if len(matches) == 1 and matches[0][0] == 0 and matches[0][1] == len(cell):
        return _resolve(matches[0][2], context), matches[0][2]
    return None, []


def normalize_value(value, spec):
    """Return a comparison value and an error; never guess ambiguous dates."""
    value = _clean(value)
    if value.casefold() in MISSING:
        return None, 'missing_value'
    kind = spec['value_type']
    if kind == 'identifier':
        if spec['name'].upper() in {'IBAN', 'BBAN', 'BIC', 'ISIN', 'LEI', 'UETR'}:
            value = re.sub(r'\s+', '', value).upper()
        return value, None
    if kind == 'currency':
        return (value.upper(), None) if re.fullmatch(r'[A-Za-z]{3}', value) else (None, 'invalid_currency')
    if kind in {'number', 'amount'}:
        value = value.replace('−', '-')
        currency = None
        code = re.fullmatch(r'(?:([A-Za-z]{3})\s+)?(.+?)(?:\s+([A-Za-z]{3}))?', value)
        if code:
            currency, value = code[1] or code[3], code[2]
        unit = '%' if value.endswith('%') else '‰' if value.endswith('‰') else ''
        if unit:
            value = value[:-1].strip()
        negative = value.startswith('(') and value.endswith(')')
        if negative:
            value = value[1:-1].strip()
        # Explicit policy: dot decimals; commas are valid only in groups of three.
        if not re.fullmatch(r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?', value):
            return None, 'invalid_number'
        try:
            number = Decimal(value.replace(',', ''))
            if negative:
                number = number.copy_abs().copy_negate()
            canonical = format(number, 'f')
            if '.' in canonical:
                canonical = canonical.rstrip('0').rstrip('.')
            if number == 0:
                canonical = '0'
            return json.dumps([canonical, unit, currency.upper() if currency else None]), None
        except InvalidOperation:
            return None, 'invalid_number'
    if kind == 'date':
        formats = ['%Y-%m-%d', '%Y/%m/%d', '%d %B %Y', '%d %b %Y', '%B %d, %Y', '%b %d, %Y']
        numeric = re.fullmatch(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', value)
        if numeric:
            a, b = int(numeric[1]), int(numeric[2])
            order = spec.get('date_order')
            if order is None and a <= 12 and b <= 12 and a != b:
                return None, 'ambiguous_date'
            order = order or ('dmy' if a > 12 or a == b else 'mdy')
            separator = '/' if '/' in value else '-'
            formats = [separator.join(('%d', '%m', '%Y') if order == 'dmy' else ('%m', '%d', '%Y'))]
        for pattern in formats:
            try:
                return datetime.strptime(value, pattern).date().isoformat(), None
            except ValueError:
                pass
        return None, 'invalid_date'
    return value.casefold(), None


def _looks_like_value(value, spec):
    """Require value-shaped text before accepting labels without delimiters."""
    value = value.strip().rstrip(';').strip()
    kind = spec['value_type']
    if kind in {'amount', 'number'}:
        return bool(re.match(r'(?:[A-Za-z]{3}\s+)?[+\-−(]?\s*\d', value))
    if kind == 'currency':
        return bool(re.fullmatch(r'[A-Za-z]{3}', value))
    if kind == 'date':
        return bool(re.match(r'\d', value)) or normalize_value(value, spec)[1] is None
    if kind == 'identifier':
        return bool(re.match(r'\d', value) or re.fullmatch(r'[A-Za-z]{1,6}\d[\w -]*', value))
    return False


def extract_field_values(text, ontology_path=DEFAULT_ONTOLOGY, input_format='plain'):
    """Extract field records with canonical identities, occurrences and evidence.

    Repeated fields are bound by occurrence in source order. Table rows also
    carry an unlabelled first-column record key when one is available.
    """
    trie = _load_schema(ontology_path)
    structured = field_text(text, input_format)
    records, issues, occurrences = [], [], Counter()
    context, headers = None, None

    def add(spec, value, line, source, record_key=None):
        canonical, error = normalize_value(value, spec)
        occurrence_key = (spec['field'], record_key)
        occurrences[occurrence_key] += 1
        records.append({'field': spec['field'], 'record': record_key,
                        'occurrence': occurrences[occurrence_key], 'value_type': spec['value_type'],
                        'raw_value': _clean(value), 'value': canonical, 'valid': error is None,
                        'error': error, 'line': line, 'source': source})

    lines = structured.splitlines()
    skip_line = None
    for line_number, raw in enumerate(lines, 1):
        if line_number == skip_line:
            continue
        line = raw.strip()
        if not line:
            continue
        if '\t' in raw:
            cells = [c.strip() for c in raw.strip(' ').split('\t')]
            labels = [_label(cell, trie, context) for cell in cells]
            if len(cells) == 2 and labels[0][1] and not labels[1][1]:
                headers = None
                if labels[0][0]:
                    add(labels[0][0], cells[1], line_number, raw)
                else:
                    issues.append({'line': line_number, 'source': raw, 'reason': 'ambiguous_field',
                                   'candidates': sorted({c['field'] for c in labels[0][1]})})
                continue
            if any(c[1] for c in labels) and all(not cell or c[1] or not re.search(r'\d', cell) for cell, c in zip(cells, labels)):
                headers = labels
                continue
            if headers and len(cells) == len(headers):
                record_key = _clean(cells[0]).casefold() if not headers[0][1] else None
                for cell, (spec, candidates) in zip(cells, headers):
                    if spec:
                        add(spec, cell, line_number, raw, record_key)
                    elif candidates:
                        issues.append({'line': line_number, 'source': raw, 'reason': 'ambiguous_field',
                                       'candidates': sorted({c['field'] for c in candidates})})
                continue
            if headers:
                issues.append({'line': line_number, 'source': raw, 'reason': 'table_column_count',
                               'expected': len(headers), 'actual': len(cells)})
                continue
        headers = None
        matches = list(_matches(line, trie))
        exact, exact_candidates = _label(line, trie, context)
        if exact_candidates and line_number < len(lines):
            following = lines[line_number].strip()
            spec = exact
            if spec and following and '\t' not in following and not list(_matches(following, trie)) and (
                    re.search(r'[:=]\s*$', line) or _looks_like_value(following, spec)):
                add(spec, following, line_number + 1, line + '\n' + following)
                skip_line = line_number + 1
                continue
        if exact and exact['owner'] is None and not re.search(r'[:=]\s*$', line):
            context = exact['field']
            continue
        previous_field = False
        # Vocabulary words inside values/prose are not automatically new labels.
        filtered = []
        for i, (start, end, candidates) in enumerate(matches):
            next_start = matches[i+1][0] if i+1 < len(matches) else len(line)
            tail = line[end:next_start]
            if re.match(r'\s*[:=]', tail) or any(_looks_like_value(tail, c) for c in candidates):
                filtered.append((start, end, candidates))
        matches = filtered
        for i, (start, end, candidates) in enumerate(matches):
            next_start = matches[i+1][0] if i+1 < len(matches) else len(line)
            tail = line[end:next_start]
            explicit = bool(re.match(r'\s*[:=]', tail))
            # Label must start a line or follow a separator / preceding typed value.
            if start and not explicit and not previous_field and not re.search(r'[;,:=\d]\s*$', line[:start]):
                continue
            spec = _resolve(candidates, context)
            if spec is None:
                if explicit or any(_looks_like_value(tail, c) for c in candidates):
                    issues.append({'line': line_number, 'source': line, 'reason': 'ambiguous_field',
                                   'candidates': sorted({c['field'] for c in candidates})})
                continue
            value = re.sub(r'^\s*[:=]\s*', '', tail).strip().rstrip(';').strip()
            if not explicit:
                # Unseparated text labels require a delimiter; typed labels may use a space.
                if not _looks_like_value(value, spec):
                    continue
            add(spec, value, line_number, line)
            previous_field = True
    # Bind a separately labelled currency to an amount in the same row/scope.
    # Occurrence alignment is conservative when repeated rows have no record key.
    currencies = [r for r in records if r['value_type'] == 'currency' and r['valid']]
    scope = lambda r: r['field'].rsplit('.', 1)[0] if '.' in r['field'] else None
    for record in records:
        if record['value_type'] != 'amount' or not record['valid']:
            continue
        value = json.loads(record['value'])
        if value[2] is not None:
            continue
        candidates = [c for c in currencies if scope(c) == scope(record) and c['record'] == record['record']]
        same_line = [c for c in candidates if c['line'] == record['line']]
        same_occurrence = [c for c in candidates if c['occurrence'] == record['occurrence']]
        candidates = same_line or same_occurrence
        if len(candidates) == 1:
            value[2] = candidates[0]['value']
            record['value'] = json.dumps(value)
            record['associated_currency'] = candidates[0]['value']
    return {'fields': records, 'issues': issues}


def field_value_metrics(left, right):
    """Occurrence-sensitive field-value Dice agreement; invalid values never match."""
    def key(record):
        return record['field'], record['record'], record['occurrence'], record['value_type'], record['value']
    a = Counter(key(r) for r in left['fields'] if r['valid'])
    b = Counter(key(r) for r in right['fields'] if r['valid'])
    shared = sum((a & b).values())
    total = len(left['fields']) + len(right['fields']) + len(left['issues']) + len(right['issues'])
    unmatched = lambda records, common: [r for r in records if not r['valid'] or key(r) not in common]
    per_field = {}
    for field in sorted({r['field'] for r in left['fields'] + right['fields']}):
        la = [r for r in left['fields'] if r['field'] == field]
        rb = [r for r in right['fields'] if r['field'] == field]
        matches = sum((Counter(key(r) for r in la if r['valid']) & Counter(key(r) for r in rb if r['valid'])).values())
        per_field[field] = {'left_count': len(la), 'right_count': len(rb), 'matched_count': matches,
                            'agreement': 2 * matches / (len(la) + len(rb))}
    return {'ontology_field_value_agreement': 2 * shared / total if total else None,
            'ontology_field_total': total, 'ontology_field_shared': shared,
            'left_ontology_fields': left['fields'], 'right_ontology_fields': right['fields'],
            'unmatched_left_ontology_fields': unmatched(left['fields'], a & b),
            'unmatched_right_ontology_fields': unmatched(right['fields'], a & b),
            'left_field_extraction_issues': left['issues'], 'right_field_extraction_issues': right['issues'],
            'per_field_agreement': per_field}


def validate_ontology_mode(mode):
    if mode not in {'regions', 'values', 'mentions'}:
        raise ValueError('ontology_mode must be regions, values or mentions.')


def primary_metric(ontology_path, mode, overlap_weight=1):
    if overlap_weight != 1:
        return 'custom_blend'
    if ontology_path is None:
        return 'token_overlap'
    if mode == 'regions':
        return 'ontology_region_agreement'
    return 'ontology_field_value_agreement' if mode == 'values' else 'ontology_weighted_token_overlap'


def primary_overlap(pair, ontology_path, mode):
    if ontology_path is not None and mode == 'regions':
        return pair['ontology_region_agreement'] or 0.0
    return ((pair['ontology_field_value_agreement'] or 0.0) if ontology_path is not None and mode == 'values'
            else pair['ontology_weighted_overlap'])
