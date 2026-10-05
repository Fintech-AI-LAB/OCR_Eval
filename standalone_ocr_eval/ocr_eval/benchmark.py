"""Reference-based critical-field OCR evaluation, independent of peer agreement.

References supply the denominator and record identity. No field is inferred from
an ontology, repaired by constraints, or removed because every engine missed it.
All metric ratios are 0--1 except CER, which may exceed one; ranking scores are
0--100. Synthetic references are always identified as demonstrations.
"""
from collections import defaultdict
from decimal import Decimal
import math
import re
import statistics
import json

from .profiles import canonical_value, fingerprint, nonempty, require, validate_profiles

BENCHMARK_VERSION = '1.0'
COUNTS = ('reference_fields', 'predicted_fields', 'detected_fields', 'correct_values',
          'literal_correct_values', 'localized_fields', 'missing_fields', 'wrong_values',
          'extra_fields', 'ignored_unreadable_predictions', 'unreadable_fields',
          'absent_fields', 'not_applicable_fields', 'critical_fields', 'critical_correct',
          'critical_weight', 'critical_correct_weight', 'field_edit_distance', 'field_characters',
          'relationships', 'correct_relationships', 'regions', 'correct_regions',
          'region_edit_distance', 'region_characters', 'eligible_records', 'correct_records',
          'eligible_documents', 'correct_documents')


def _box(value, where):
    require(isinstance(value, list) and len(value) == 4
            and all(type(x) in {int, float} and math.isfinite(x) and 0 <= x <= 1 for x in value)
            and value[0] < value[2] and value[1] < value[3],
            f'{where}: bbox must be normalized [left, top, right, bottom] with positive area.')


def _location(item, where, required=False):
    if required or item.get('page') is not None:
        require(type(item.get('page')) is int and item['page'] > 0, f'{where}: page must be a positive integer.')
    if required or item.get('bbox') is not None:
        _box(item.get('bbox'), where)
        require(type(item.get('page')) is int and item['page'] > 0, f'{where}: bbox requires page.')


def validate_references(references, profiles):
    validate_profiles(profiles)
    require(isinstance(references, dict) and type(references.get('schema_version')) is int
            and references['schema_version'] == 1, 'References require schema_version: 1.')
    status = references.get('annotation_status')
    require(status in {'verified', 'synthetic'},
            'References must be image-verified or explicitly synthetic; draft annotations cannot be scored.')
    documents = references.get('documents')
    require(isinstance(documents, dict) and documents, 'References require a nonempty documents mapping.')
    template_splits = {}
    critical_count = 0
    for document, ref in documents.items():
        require(nonempty(document) and isinstance(ref, dict), 'Invalid reference document.')
        require(ref.get('profile') in profiles['profiles'], f'{document}: unknown profile.')
        profile = profiles['profiles'][ref['profile']]
        if status == 'verified':
            require(profile['review_status'] == 'reviewed', f'{document}: verified evaluation requires a reviewed profile.')
            require(nonempty(ref.get('reviewed_by')), f'{document}: reviewed_by is required.')
        require(nonempty(ref.get('template_id')), f'{document}: template_id required.')
        require(ref.get('split') in {'train', 'validation', 'test'}, f'{document}: invalid split.')
        previous = template_splits.setdefault(ref['template_id'], ref['split'])
        require(previous == ref['split'], f'Template {ref["template_id"]} occurs in multiple splits.')
        source = ref.get('source')
        require(isinstance(source, dict) and nonempty(source.get('path'))
                and isinstance(source.get('sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', source['sha256']),
                f'{document}: source path and SHA-256 required.')
        records = ref.get('records')
        require(isinstance(records, list) and records, f'{document}: nonempty records required.')
        seen = set()
        for record in records:
            require(isinstance(record, dict) and nonempty(record.get('id')), f'{document}: record id required.')
            require(record['id'] not in seen, f'{document}: duplicate reference record {record["id"]}.')
            seen.add(record['id'])
            require(isinstance(record.get('fields'), dict) and set(record['fields']) == set(profile['fields']),
                    f'{document}/{record["id"]}: annotate every profile field, including absent/not_applicable fields.')
            for field, annotation in record['fields'].items():
                where = f'{document}/{record["id"]}/{field}'
                require(isinstance(annotation, dict) and annotation.get('status') in
                        {'present', 'absent', 'unreadable', 'not_applicable'}, f'{where}: invalid annotation status.')
                if annotation['status'] == 'present':
                    require(nonempty(annotation.get('value')), f'{where}: present fields require a literal transcription.')
                    spec = profile['fields'][field]
                    require(canonical_value(annotation['value'], field, spec)[1] is None,
                            f'{where}: reference value cannot be normalized; check type/date_order or use literal comparison.')
                    _location(annotation, where, required=True)
                    critical_count += int(spec['critical'])
                else:
                    require(annotation.get('value') is None, f'{where}: non-present fields must not contain a value.')
                    if annotation['status'] == 'unreadable':
                        _location(annotation, where, required=True)
        regions = ref.get('regions', [])
        require(isinstance(regions, list), f'{document}: regions must be a list.')
        ids = set()
        for region in regions:
            require(isinstance(region, dict) and nonempty(region.get('id')) and region['id'] not in ids,
                    f'{document}: duplicate or invalid reference region.')
            ids.add(region['id'])
            require(nonempty(region.get('text')), f'{document}: reference region text must be nonempty.')
            _location(region, f'{document}/{region["id"]}', required=True)
    require(critical_count > 0, 'References contain no readable critical fields.')
    return references


def _validate_predictions(predictions, references, profiles):
    require(isinstance(predictions, dict) and type(predictions.get('schema_version')) is int
            and predictions['schema_version'] == 1, 'Predictions require schema_version: 1.')
    if predictions.get('profile_sha256') is not None:
        require(predictions['profile_sha256'] == fingerprint(profiles), 'Prediction extraction used a different profile.')
    methods = predictions.get('methods')
    require(isinstance(methods, dict) and methods, 'Predictions require at least one method.')
    for method, data in methods.items():
        require(nonempty(method) and isinstance(data, dict) and nonempty(data.get('family')),
                'Each method needs a name and engine family.')
        docs = data.get('documents')
        require(isinstance(docs, dict) and set(docs) == set(references['documents']),
                f'{method}: explicitly supply every reference document; use fields: [] for failed/empty output.')
        for document, prediction in docs.items():
            require(isinstance(prediction, dict) and isinstance(prediction.get('fields'), list),
                    f'{method}/{document}: fields must be a list.')
            ref = references['documents'][document]
            if prediction.get('source_sha256') is not None:
                require(prediction['source_sha256'] == ref['source']['sha256'], f'{method}/{document}: source hash mismatch.')
            for item in prediction['fields']:
                require(isinstance(item, dict) and nonempty(item.get('field')) and nonempty(item.get('record'))
                        and isinstance(item.get('value'), str), f'{method}/{document}: invalid predicted field.')
                _location(item, f'{method}/{document}/{item["field"]}')
            regions = prediction.get('regions', {})
            require(isinstance(regions, dict) and all(isinstance(v, str) for v in regions.values()),
                    f'{method}/{document}: regions must map reference region IDs to transcriptions.')
            require(set(regions) <= {r['id'] for r in ref.get('regions', [])},
                    f'{method}/{document}: unknown reference region ID.')


def edit_distance(a, b):
    """Exact character Levenshtein distance with linear auxiliary space."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def _iou(a, b):
    if a.get('page') != b.get('page') or a.get('bbox') is None or b.get('bbox') is None:
        return 0.0
    x, y = a['bbox'], b['bbox']
    intersection = max(0, min(x[2], y[2]) - max(x[0], y[0])) * max(0, min(x[3], y[3]) - max(x[1], y[1]))
    area = (x[2]-x[0]) * (x[3]-x[1]) + (y[2]-y[0]) * (y[3]-y[1])
    return intersection / (area-intersection)


def constraint_violations(fields, profile, record_ids=None, excluded_keys=()):
    """Report schema plausibility separately; never correct or score from it."""
    grouped = defaultdict(list)
    for item in fields:
        grouped[item['record'], item['field']].append(item)
    records = set(record_ids or ()) | {r for r, _ in grouped}
    issues = []
    for record in sorted(records):
        for field, spec in profile['fields'].items():
            if (record, field) in excluded_keys:
                continue
            items = grouped[record, field]
            def issue(reason, **extra):
                issues.append({'record': record, 'field': field, 'reason': reason, **extra})
            if spec.get('required', False) and not any(i['value'].strip() for i in items):
                issue('required_field_missing')
            if len(items) > 1:
                issue('duplicate_field', count=len(items))
            for item in items:
                value, error = canonical_value(item['value'], field, spec)
                if error:
                    issue(error, raw_value=item['value'])
                rules = spec.get('constraints', {})
                if 'pattern' in rules and re.fullmatch(rules['pattern'], item['value']) is None:
                    issue('pattern', raw_value=item['value'])
                if 'enum' in rules and value not in {canonical_value(v, field, spec)[0] for v in rules['enum']}:
                    issue('enum', raw_value=item['value'])
                # Bounds use the existing typed numeric policy even for literal comparison.
                typed, invalid = canonical_value(item['value'], field, {**spec, 'comparison': 'typed'})
                if not invalid and spec['value_type'] in {'amount', 'number'}:
                    number = Decimal(json.loads(typed)[0])
                    for bound in ('minimum', 'maximum'):
                        if bound in rules and ((bound == 'minimum' and number < Decimal(str(rules[bound])))
                                               or (bound == 'maximum' and number > Decimal(str(rules[bound])))):
                            issue(bound, raw_value=item['value'])
                for target in spec.get('requires', []):
                    if (record, target) not in excluded_keys and not any(i['value'].strip() for i in grouped[record, target]):
                        issue('missing_relationship_target', target=target)
    for item in fields:
        if item['field'] not in profile['fields']:
            issues.append({'record': item['record'], 'field': item['field'], 'reason': 'unknown_field'})
    return issues


def _counts():
    return {key: 0 for key in COUNTS}


def _sum_counts(items):
    items = list(items)
    return {key: math.fsum(item[key] for item in items) if key.endswith('weight')
            else sum(item[key] for item in items) for key in COUNTS}


def _ratio(a, b):
    return a / b if b else None


def _metrics(c):
    return {**c,
            'field_precision': _ratio(c['detected_fields'], c['predicted_fields']),
            'field_recall': _ratio(c['detected_fields'], c['reference_fields']),
            'value_precision': _ratio(c['correct_values'], c['predicted_fields']),
            'value_recall': _ratio(c['correct_values'], c['reference_fields']),
            'value_f1': _ratio(2*c['correct_values'], c['predicted_fields']+c['reference_fields']),
            'literal_value_accuracy': _ratio(c['literal_correct_values'], c['reference_fields']),
            'critical_field_accuracy': _ratio(c['critical_correct'], c['critical_fields']),
            'weighted_critical_field_accuracy': _ratio(c['critical_correct_weight'], c['critical_weight']),
            'localization_recall': _ratio(c['localized_fields'], c['reference_fields']),
            'field_character_error_rate': _ratio(c['field_edit_distance'], c['field_characters']),
            'relationship_accuracy': _ratio(c['correct_relationships'], c['relationships']),
            'record_accuracy': _ratio(c['correct_records'], c['eligible_records']),
            'all_critical_fields_correct_document_rate': _ratio(c['correct_documents'], c['eligible_documents']),
            'critical_region_character_error_rate': _ratio(c['region_edit_distance'], c['region_characters']),
            'critical_region_exact_accuracy': _ratio(c['correct_regions'], c['regions'])}


def _score_document(ref, prediction, profile, iou_threshold):
    c = _counts()
    by_field = defaultdict(_counts)
    annotations = {(record['id'], f): item for record in ref['records'] for f, item in record['fields'].items()}
    present = {key: item for key, item in annotations.items() if item['status'] == 'present'}
    predictions = prediction['fields']
    groups = defaultdict(list)
    ignored, used, correct = set(), set(), set()
    matches, errors = [], []
    for index, item in enumerate(predictions):
        key = item['record'], item['field']
        if key in annotations and annotations[key]['status'] == 'unreadable':
            ignored.add(index)
            c['ignored_unreadable_predictions'] += 1
            by_field[item['field']]['ignored_unreadable_predictions'] += 1
        else:
            groups[key].append(index)
            c['predicted_fields'] += 1
            by_field[item['field']]['predicted_fields'] += 1
    for (record, field), annotation in annotations.items():
        if annotation['status'] != 'present':
            counter = {'absent': 'absent_fields', 'unreadable': 'unreadable_fields',
                       'not_applicable': 'not_applicable_fields'}[annotation['status']]
            c[counter] += 1
            by_field[field][counter] += 1
    for key, annotation in present.items():
        record, field = key
        spec = profile['fields'][field]
        reference_value = canonical_value(annotation['value'], field, spec)[0]

        def quality(index):
            item = predictions[index]
            value, error = canonical_value(item['value'], field, spec)
            return (error is None and value == reference_value, item['value'] == annotation['value'],
                    _iou(item, annotation), -index)

        candidates = groups[key]
        index = max(candidates, key=quality) if candidates else None
        item = predictions[index] if index is not None else None
        detected = item is not None and bool(item['value'].strip())
        same = bool(detected and quality(index)[0])
        literal = bool(detected and item['value'] == annotation['value'])
        localized = bool(detected and _iou(item, annotation) >= iou_threshold)
        if detected:
            used.add(index)
        if same:
            correct.add(key)
        edits = edit_distance(annotation['value'], item['value'] if detected else '')
        increments = {'reference_fields': 1, 'detected_fields': int(detected), 'correct_values': int(same),
                      'literal_correct_values': int(literal), 'localized_fields': int(localized),
                      'missing_fields': int(not detected), 'wrong_values': int(detected and not same),
                      'field_edit_distance': edits, 'field_characters': len(annotation['value']),
                      'critical_fields': int(spec['critical']), 'critical_correct': 0,
                      'critical_weight': spec['weight'] if spec['critical'] else 0,
                      'critical_correct_weight': 0}
        for counter, value in increments.items():
            c[counter] += value
            by_field[field][counter] += value
        matches.append({'record': record, 'field': field, 'reference_value': annotation['value'],
                        'predicted_value': item['value'] if item is not None else None,
                        'reference_location': {k: annotation[k] for k in ('page', 'bbox')},
                        'predicted_location': {k: item.get(k) for k in ('page', 'bbox')} if item else None,
                        'prediction_index': index, 'correct': same, 'literal_correct': literal,
                        'localized': localized, 'character_errors': edits,
                        'reason': 'correct' if same else 'wrong_value' if detected else 'missing'})
    extras = []
    for index, item in enumerate(predictions):
        if index in used or index in ignored:
            continue
        field = item['field']
        c['extra_fields'] += 1
        by_field[field]['extra_fields'] += 1
        c['field_edit_distance'] += len(item['value'])
        by_field[field]['field_edit_distance'] += len(item['value'])
        reason = 'duplicate' if (item['record'], field) in present else 'extra'
        if field in profile['fields']:
            value, error = canonical_value(item['value'], field, profile['fields'][field])
            if error is None and any(other_field == field and other_record != item['record']
                                     and canonical_value(a['value'], field, profile['fields'][field])[0] == value
                                     for (other_record, other_field), a in present.items()):
                reason = 'wrong_record'
        extras.append(index)
        errors.append({'prediction_index': index, **item, 'reason': reason})
    relationships = []
    for (record, field), annotation in present.items():
        for target in profile['fields'][field].get('requires', []):
            if (record, target) not in present:
                continue
            ok = (record, field) in correct and (record, target) in correct
            c['relationships'] += 1
            c['correct_relationships'] += int(ok)
            relationships.append({'record': record, 'field': field, 'target': target, 'correct': ok})
    # Credit a critical field only when its value and each declared, readable
    # associated value in the same record are correct. Raw value accuracy above
    # remains independent of these relationships.
    complete_correct = {key for key in correct if all(
        (key[0], target) in correct for target in profile['fields'][key[1]].get('requires', [])
        if (key[0], target) in present)}
    for key in complete_correct:
        field = key[1]
        spec = profile['fields'][field]
        if spec['critical']:
            c['critical_correct'] += 1
            c['critical_correct_weight'] += spec['weight']
            by_field[field]['critical_correct'] += 1
            by_field[field]['critical_correct_weight'] += spec['weight']
    for match in matches:
        match['value_and_relationships_correct'] = (match['record'], match['field']) in complete_correct
    critical_scope = {f for f, spec in profile['fields'].items() if spec['critical']}
    critical_scope.update(target for f, spec in profile['fields'].items() if spec['critical']
                          for target in spec.get('requires', []))
    critical_extras = [predictions[i] for i in extras
                       if predictions[i]['field'] not in profile['fields']
                       or predictions[i]['field'] in critical_scope]
    unreadable_critical = any(a['status'] == 'unreadable' and f in critical_scope
                              for (_, f), a in annotations.items())
    for record in ref['records']:
        keys = {key for key in present if key[0] == record['id'] and profile['fields'][key[1]]['critical']}
        unreadable = any(a['status'] == 'unreadable' and f in critical_scope
                         for f, a in record['fields'].items())
        if keys and not unreadable:
            c['eligible_records'] += 1
            c['correct_records'] += int(keys <= complete_correct and not any(e['record'] == record['id'] for e in critical_extras))
    if c['critical_fields'] and not unreadable_critical:
        c['eligible_documents'] = 1
        c['correct_documents'] = int(c['critical_correct'] == c['critical_fields'] and not critical_extras)
    for region in ref.get('regions', []):
        text = prediction.get('regions', {}).get(region['id'], '')
        c['regions'] += 1
        c['correct_regions'] += int(text == region['text'])
        c['region_characters'] += len(region['text'])
        c['region_edit_distance'] += edit_distance(region['text'], text)
    source_fields = [{'record': record, 'field': field, 'value': a['value']} for (record, field), a in present.items()]
    record_ids = [r['id'] for r in ref['records']]
    excluded_keys = {key for key, a in annotations.items() if a['status'] in {'not_applicable', 'unreadable'}}
    return {'counts': c, 'metrics': _metrics(c), 'per_field': {f: _metrics(v) for f, v in by_field.items()},
            'field_results': matches, 'extra_predictions': errors, 'relationships': relationships,
            'constraint_violations': constraint_violations(predictions, profile, record_ids, excluded_keys),
            'source_constraint_violations': constraint_violations(source_fields, profile, record_ids, excluded_keys),
            'extraction_issues': prediction.get('extraction_issues', []),
            'document_pass_ineligible_reason': 'unreadable critical fields or required association targets' if unreadable_critical
                else 'no present critical fields' if not c['critical_fields'] else None,
            'source_hash_checked': prediction.get('source_sha256') is not None}


def _rank(methods):
    ordered = sorted(methods, key=lambda m: (-methods[m]['score'], m))
    rows, previous, rank = [], None, 0
    for position, method in enumerate(ordered, 1):
        score = methods[method]['score']
        if previous is None or not math.isclose(previous, score, rel_tol=0, abs_tol=1e-10):
            rank = position
        rows.append({'rank': rank, 'method': method, 'critical_field_score': score})
        previous = score
    return rows


def evaluate_benchmark(profiles, references, predictions, *, split='test', weighting='document', iou_threshold=0.5):
    """Score submitted field records and fixed-region transcripts against references."""
    validate_references(references, profiles)
    _validate_predictions(predictions, references, profiles)
    require(split in {'train', 'validation', 'test', 'all'}, 'split must be train, validation, test or all.')
    require(weighting in {'document', 'field'}, 'weighting must be document or field.')
    require(type(iou_threshold) in {int, float} and math.isfinite(iou_threshold) and 0 < iou_threshold <= 1,
            'iou_threshold must be finite and in (0, 1].')
    documents = {d: r for d, r in references['documents'].items() if split == 'all' or r['split'] == split}
    require(documents, f'No reference documents in split {split}.')
    results = {}
    for method, data in predictions['methods'].items():
        docs = {d: _score_document(r, data['documents'][d], profiles['profiles'][r['profile']], iou_threshold)
                for d, r in documents.items()}
        total = _sum_counts(v['counts'] for v in docs.values())
        require(total['critical_fields'] > 0, f'No readable critical fields in split {split}.')
        metrics = _metrics(total)
        scores = [v['metrics']['weighted_critical_field_accuracy'] for v in docs.values()
                  if v['metrics']['weighted_critical_field_accuracy'] is not None]
        score = statistics.mean(scores) if weighting == 'document' else metrics['weighted_critical_field_accuracy']
        fields = sorted({f for v in docs.values() for f in v['per_field']})
        per_field = {f: _metrics(_sum_counts({k: v['per_field'][f][k] for k in COUNTS}
                     for v in docs.values() if f in v['per_field'])) for f in fields}
        templates = sorted({r['template_id'] for r in documents.values()})
        per_template = {t: _metrics(_sum_counts(v['counts'] for d, v in docs.items()
                                              if documents[d]['template_id'] == t)) for t in templates}
        results[method] = {'family': data['family'], 'provenance': data.get('provenance', {}),
                           'score': 100*score, 'metrics': metrics,
                           'documents': docs, 'per_field': per_field, 'per_template': per_template}
    return {'benchmark_version': BENCHMARK_VERSION, 'annotation_status': references['annotation_status'],
            'meaning': 'Synthetic demonstration; not measured OCR performance.' if references['annotation_status'] == 'synthetic'
                       else 'Accuracy against supplied image-verified annotations; verification is declared by the annotator.',
            'settings': {'split': split, 'weighting': weighting, 'iou_threshold': iou_threshold,
                         'primary_metric': 'weighted_critical_field_accuracy',
                         'normalization': 'Profile comparison policy; raw values retained.',
                         'document_ids': list(documents), 'profile_sha256': fingerprint(profiles),
                         'reference_sha256': fingerprint(references), 'predictions_sha256': fingerprint(predictions)},
            'reference_documents': {d: {k: r.get(k) for k in ('profile', 'template_id', 'split', 'source', 'reviewed_by')}
                                    for d, r in documents.items()},
            'ranking': _rank(results), 'methods': results,
            'documents_without_readable_critical_fields': [d for d, r in documents.items() if not any(
                a['status'] == 'present' and profiles['profiles'][r['profile']]['fields'][f]['critical']
                for record in r['records'] for f, a in record['fields'].items())]}


def compare_benchmark_variants(profiles, references, variants, **settings):
    """Compare submitted extraction variants with fixed references, profiles and metrics.

    Supply independently generated predictions for each variant. This function
    does not claim to train an extractor or generate experimental evidence.
    """
    require(isinstance(variants, dict) and len(variants) >= 2, 'Supply at least two named prediction variants.')
    results, methods = {}, None
    for name, predictions in variants.items():
        require(nonempty(name), 'Variant names must be nonempty strings.')
        result = evaluate_benchmark(profiles, references, predictions, **settings)
        current = set(result['methods'])
        methods = current if methods is None else methods
        require(methods == current, 'All variants must contain the same methods.')
        results[name] = result
    baseline = next(iter(results))
    differences = {name: {m: result['methods'][m]['score'] - results[baseline]['methods'][m]['score']
                          for m in methods} for name, result in results.items()}
    return {'baseline': baseline, 'score_differences': differences, 'variants': results}


def consensus_family_sensitivity(pairs, families):
    """Diagnose peer dependence from existing pair results without rerunning OCR.

    Average pages within documents and documents equally, then peers within each
    other engine family and those families equally. Exclude same-family peers.
    """
    require(isinstance(pairs, list) and pairs and isinstance(families, dict) and families
            and all(nonempty(k) and nonempty(v) for k, v in families.items()), 'Pairs and engine families required.')
    groups, seen = defaultdict(list), set()
    units = defaultdict(set)
    for row in pairs:
        a, b, document, page = row.get('left'), row.get('right'), row.get('document'), row.get('page')
        score = row.get('primary_similarity', row.get('combined_similarity'))
        require(a in families and b in families and a != b and nonempty(document)
                and type(page) is int and page > 0, 'Invalid pair identity or missing engine family.')
        require(type(score) in {int, float} and math.isfinite(score) and 0 <= score <= 1,
                'Pair scores must be finite in [0, 1].')
        key = document, page, *sorted((a, b))
        require(key not in seen, 'Duplicate pair observation.')
        seen.add(key)
        units[document, page].update((a, b))
        groups[tuple(sorted((a, b))), document].append(score)
    from itertools import combinations
    for document, page in units:
        require(all((document, page, a, b) in seen for a, b in combinations(sorted(families), 2)),
                'Family sensitivity requires a complete peer matrix on every supplied page.')
    pair_scores = {pair: statistics.mean(statistics.mean(values) for (p, _), values in groups.items() if p == pair)
                   for pair, _ in groups}

    def scores(excluded=None):
        result = {}
        for method, family in families.items():
            if family == excluded:
                continue
            peers = defaultdict(list)
            for other, other_family in families.items():
                if other_family not in {family, excluded}:
                    peers[other_family].append(pair_scores[tuple(sorted((method, other)))])
            result[method] = 100*statistics.mean(statistics.mean(v) for v in peers.values()) if peers else None
        return result

    return {'meaning': 'Consensus sensitivity, not correctness.', 'family_balanced_scores': scores(),
            'leave_one_family_out': {f: scores(f) for f in sorted(set(families.values()))}}
