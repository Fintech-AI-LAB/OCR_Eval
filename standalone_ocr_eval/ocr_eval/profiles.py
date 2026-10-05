"""Reviewed document profiles and conservative, reference-independent extraction."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re

from .field_values import VALUE_TYPES, _variants, field_text, normalize_value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def load_json(path):
    """Reject duplicate keys and non-finite numbers instead of losing evidence."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f'Non-finite JSON number: {value}')

    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=unique, parse_constant=invalid)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def canonical_value(value, field, spec):
    if spec.get('comparison', 'typed') == 'literal':
        return (value, None) if value.strip() else (None, 'missing_value')
    return normalize_value(value, {'value_type': spec['value_type'],
                                   'name': field.rsplit('.', 1)[-1],
                                   'date_order': spec.get('date_order')})


def validate_profiles(bundle):
    require(isinstance(bundle, dict) and type(bundle.get('schema_version')) is int
            and bundle['schema_version'] == 1, 'Profiles require schema_version: 1.')
    profiles = bundle.get('profiles')
    require(isinstance(profiles, dict) and profiles, 'Profiles must be a nonempty mapping.')
    for name, profile in profiles.items():
        require(nonempty(name) and isinstance(profile, dict), 'Invalid document profile.')
        require(profile.get('review_status') in {'draft', 'example', 'reviewed'},
                f'{name}: specify review_status (draft, example, reviewed).')
        if profile['review_status'] == 'reviewed':
            require(nonempty(profile.get('reviewed_by')), f'{name}: reviewed_by is required.')
        questions = profile.get('competency_questions')
        require(isinstance(questions, list) and questions and all(map(nonempty, questions)),
                f'{name}: competency_questions must describe the intended task.')
        fields = profile.get('fields')
        require(isinstance(fields, dict) and fields, f'{name}: fields must be a nonempty mapping.')
        require(any(isinstance(s, dict) and s.get('critical') is True for s in fields.values()),
                f'{name}: at least one critical field is required.')
        for field, spec in fields.items():
            require(nonempty(field) and isinstance(spec, dict), f'{name}: invalid field specification.')
            allowed = {'aliases', 'value_type', 'comparison', 'date_order', 'critical', 'weight',
                       'rationale', 'required', 'requires', 'constraints', 'ontology_id'}
            require(not set(spec) - allowed, f'{name}/{field}: unknown field settings: {set(spec) - allowed}')
            require(spec.get('value_type') in VALUE_TYPES, f'{field}: specify a supported value_type.')
            require(type(spec.get('critical')) is bool, f'{field}: critical must be a boolean.')
            require(type(spec.get('required', False)) is bool, f'{field}: required must be a boolean.')
            weight = spec.get('weight')
            require(type(weight) in {int, float} and math.isfinite(weight) and weight > 0,
                    f'{field}: weight must be finite and positive.')
            require(nonempty(spec.get('rationale')), f'{field}: an error-cost rationale is required.')
            aliases = spec.get('aliases', [])
            require(isinstance(aliases, list) and all(map(nonempty, aliases)), f'{field}: invalid aliases.')
            require(spec.get('comparison', 'typed') in {'typed', 'literal'}, f'{field}: invalid comparison.')
            require(spec.get('date_order') in {None, 'dmy', 'mdy'}, f'{field}: invalid date_order.')
            links = spec.get('requires', [])
            require(isinstance(links, list) and all(nonempty(x) and x in fields and x != field for x in links)
                    and len(set(links)) == len(links), f'{field}: requires must name distinct other profile fields.')
            constraints = spec.get('constraints', {})
            require(isinstance(constraints, dict) and not set(constraints) - {'pattern', 'enum', 'minimum', 'maximum'},
                    f'{field}: unsupported constraints.')
            if 'pattern' in constraints:
                require(isinstance(constraints['pattern'], str), f'{field}: pattern must be a string.')
                try:
                    re.compile(constraints['pattern'])
                except re.error as error:
                    raise ValueError(f'{field}: invalid pattern: {error}') from error
            if 'enum' in constraints:
                values = constraints['enum']
                require(isinstance(values, list) and values and all(map(nonempty, values)), f'{field}: invalid enum.')
                require(all(canonical_value(v, field, spec)[1] is None for v in values), f'{field}: invalid typed enum.')
            for bound in ('minimum', 'maximum'):
                if bound in constraints:
                    require(spec['value_type'] in {'amount', 'number'}, f'{field}: bounds require a numeric type.')
                    try:
                        number = Decimal(str(constraints[bound]))
                        require(number.is_finite(), f'{field}: bound must be finite.')
                    except InvalidOperation as error:
                        raise ValueError(f'{field}: invalid numeric bound.') from error
            if 'minimum' in constraints and 'maximum' in constraints:
                require(Decimal(str(constraints['minimum'])) <= Decimal(str(constraints['maximum'])),
                        f'{field}: minimum exceeds maximum.')
        keys = profile.get('record_key_labels', ['Record', 'Item', 'Transaction ID'])
        require(isinstance(keys, list) and keys and all(map(nonempty, keys)), f'{name}: invalid record_key_labels.')
        labels = profile_labels(profile)
        require(not set(labels) & {label_key(k) for k in keys}, f'{name}: record key labels overlap field labels.')
    return bundle


def load_profiles(path):
    return validate_profiles(load_json(path))


def label_key(label):
    return ' '.join(re.findall(r'[^\W_]+', label.casefold()))


def profile_labels(profile):
    labels = {}
    for field, spec in profile['fields'].items():
        for name in [field, *spec.get('aliases', [])]:
            for variant in _variants(name):
                label = ' '.join(variant)
                require(label not in labels or labels[label] == field,
                        f'Ambiguous profile alias {name!r}; use scoped labels or separate profiles.')
                labels[label] = field
    return labels


def extract_profile_fields(pages, profile, input_format='plain'):
    """Extract explicit pairs and keyed tables without access to reference values.

    Stable table keys come from declared record-key columns. Unkeyed fields use
    record ``document``; duplicates remain duplicates rather than being inferred.
    """
    require(isinstance(pages, list) and pages and all(isinstance(p, str) for p in pages),
            'pages must be a nonempty list of strings.')
    labels = profile_labels(profile)
    key_labels = {label_key(k) for k in profile.get('record_key_labels', ['Record', 'Item', 'Transaction ID'])}
    fields, issues = [], []
    for page, text in enumerate(pages, 1):
        lines = field_text(text, input_format).splitlines()
        header, key_column, skip = None, None, False

        def add(field, value, line, record='document'):
            fields.append({'field': field, 'record': record, 'value': value.strip(),
                           'page': page, 'source_line': line})

        for line_number, line in enumerate(lines, 1):
            if skip:
                skip = False
                continue
            if not line.strip():
                continue
            if '\t' in line:
                cells = [c.strip() for c in line.split('\t')]
                names = [label_key(c.rstrip(':=')) for c in cells]
                key_columns = [i for i, c in enumerate(names) if c in key_labels]
                if key_columns and any(c in labels for c in names):
                    require(len(key_columns) == 1, f'Page {page}, line {line_number}: ambiguous record key columns.')
                    header, key_column = [labels.get(c) for c in names], key_columns[0]
                    continue
                if header is not None:
                    if len(cells) != len(header) or not cells[key_column]:
                        issues.append({'page': page, 'line': line_number, 'reason': 'invalid_table_row', 'text': line})
                    else:
                        for field, value in zip(header, cells):
                            if field:
                                add(field, value, line_number, cells[key_column])
                    continue
                if len(cells) == 2 and names[0] in labels and names[1] not in labels:
                    add(labels[names[0]], cells[1], line_number)
                elif any(c in labels for c in names):
                    issues.append({'page': page, 'line': line_number, 'reason': 'table_requires_record_key', 'text': line})
                continue
            header, key_column = None, None
            pair = re.match(r'^\s*(.*?)\s*[:=]\s*(.*)$', line)
            if pair and label_key(pair[1]) in labels:
                field = labels[label_key(pair[1])]
                value = pair[2]
                if not value and line_number < len(lines):
                    following = lines[line_number].strip()
                    if following and '\t' not in following and not re.search(r'[:=]', following) and label_key(following) not in labels:
                        value, skip = following, True
                add(field, value, line_number)
            elif label_key(line) in labels and line_number < len(lines):
                following = lines[line_number].strip()
                field = labels[label_key(line)]
                if following and '\t' not in following and not re.search(r'[:=]', following) and label_key(following) not in labels:
                    add(field, following, line_number)
                    skip = True
    return {'fields': fields, 'regions': {}, 'extraction_issues': issues}


def extract_profile_predictions(data, profiles, document_profiles, families=None):
    """Adapt the existing aligned-page JSON to benchmark prediction records."""
    validate_profiles(profiles)
    require(isinstance(data, dict) and data and set(data) == set(document_profiles),
            'Document/profile mapping must cover exactly the input documents.')
    methods, expected = {}, None
    for document, records in data.items():
        require(isinstance(records, dict) and records, f'{document}: no methods supplied.')
        expected = set(records) if expected is None else expected
        require(set(records) == expected, f'{document}: method coverage differs.')
        name = document_profiles[document]
        require(name in profiles['profiles'], f'{document}: unknown profile {name}.')
        for method, record in records.items():
            require(nonempty(method) and isinstance(record, dict), 'Invalid method record.')
            methods.setdefault(method, {'family': (families or {}).get(method, method), 'documents': {}})
            result = extract_profile_fields(record.get('pages'), profiles['profiles'][name], record.get('input_format', 'markdown'))
            result['source_sha256'] = record.get('source_sha256')
            methods[method]['documents'][document] = result
    return {'schema_version': 1, 'profile_sha256': fingerprint(profiles), 'methods': methods,
            'extraction_policy': 'Explicit label/value pairs and tables with declared stable record keys; no reference access.'}


def make_reference_template(profiles, manifest, source_root='.'):
    """Create a deliberately unscorable annotation inventory from source files."""
    validate_profiles(profiles)
    require(isinstance(manifest, dict) and isinstance(manifest.get('documents'), dict)
            and manifest['documents'], 'Manifest requires a nonempty documents mapping.')
    result = {'schema_version': 1, 'annotation_status': 'draft', 'documents': {}}
    for document, item in manifest['documents'].items():
        require(isinstance(item, dict) and item.get('profile') in profiles['profiles'], f'{document}: unknown profile.')
        require(nonempty(item.get('source_path')), f'{document}: source_path required.')
        source = Path(source_root) / item['source_path']
        ids = item.get('record_ids', ['document'])
        require(isinstance(ids, list) and ids and all(map(nonempty, ids)) and len(set(ids)) == len(ids),
                f'{document}: record_ids must be distinct nonempty strings.')
        require(nonempty(item.get('template_id')), f'{document}: template_id required.')
        result['documents'][document] = {
            'profile': item['profile'], 'template_id': item['template_id'],
            'split': item.get('split', 'test'), 'reviewed_by': '',
            'source': {'path': str(source.resolve()), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
            'records': [{'id': key, 'fields': {f: {'status': 'unannotated', 'value': None, 'page': None, 'bbox': None}
                         for f in profiles['profiles'][item['profile']]['fields']}} for key in ids],
            'regions': []}
    return deepcopy(result)
