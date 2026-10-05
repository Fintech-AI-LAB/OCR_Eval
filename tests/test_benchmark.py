import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone_ocr_eval'))
from ocr_eval import (evaluate_benchmark, compare_benchmark_variants, consensus_family_sensitivity,
                      extract_profile_predictions, make_reference_template)
from ocr_eval.profiles import validate_profiles, extract_profile_fields, load_json
from ocr_eval.benchmark import edit_distance
from ocr_eval.cli import main


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        def spec(kind, weight=1, **extra):
            return {'value_type': kind, 'critical': True, 'weight': weight,
                    'rationale': 'Synthetic test consequence', **extra}
        self.profiles = {'schema_version': 1, 'profiles': {'trade': {
            'review_status': 'example', 'competency_questions': ['Which record receives which amount?'],
            'fields': {'Amount': spec('amount', 5, aliases=['Settlement Amount'], requires=['Currency']),
                       'Currency': spec('currency', constraints={'enum': ['USD', 'EUR']}),
                       'Account': spec('identifier', aliases=['Account No'], required=True)}}}}
        self.box = [0.1, 0.1, 0.3, 0.2]
        self.references = {'schema_version': 1, 'annotation_status': 'synthetic', 'documents': {'doc': {
            'profile': 'trade', 'template_id': 'template-a', 'split': 'test',
            'source': {'path': 'synthetic.txt', 'sha256': '0'*64},
            'records': [{'id': 'A', 'fields': {field: {'status': 'present', 'value': value,
                         'page': 1, 'bbox': list(self.box)} for field, value in
                         [('Amount', '100.00'), ('Currency', 'USD'), ('Account', '00123')]}}],
            'regions': [{'id': 'region-1', 'page': 1, 'bbox': list(self.box), 'text': '100.00 USD 00123'}]}}}
        self.predictions = {'schema_version': 1, 'methods': {'engine': {'family': 'independent-a', 'documents': {
            'doc': {'fields': [{'record': 'A', 'field': f, 'value': a['value'], 'page': a['page'], 'bbox': a['bbox']}
                              for f, a in self.references['documents']['doc']['records'][0]['fields'].items()],
                    'regions': {'region-1': '100.00 USD 00123'}, 'source_sha256': '0'*64}}}}}

    @property
    def fields(self):
        return self.predictions['methods']['engine']['documents']['doc']['fields']

    def run_benchmark(self, **kwargs):
        return evaluate_benchmark(self.profiles, self.references, self.predictions, **kwargs)

    def metrics(self, **kwargs):
        return self.run_benchmark(**kwargs)['methods']['engine']['metrics']

    def test_perfect_reference_score_and_provenance(self):
        result = self.run_benchmark()
        self.assertEqual(result['ranking'][0]['critical_field_score'], 100)
        for metric in ('value_precision', 'value_recall', 'localization_recall', 'relationship_accuracy',
                       'record_accuracy', 'all_critical_fields_correct_document_rate', 'critical_region_exact_accuracy'):
            self.assertEqual(self.metrics()[metric], 1, metric)
        self.assertEqual(self.metrics()['critical_region_character_error_rate'], 0)
        self.assertIn('Synthetic demonstration', result['meaning'])
        self.assertEqual(len(result['settings']['reference_sha256']), 64)
        self.assertEqual(result['methods']['engine']['documents']['doc']['constraint_violations'], [])

    def test_changed_amount_has_full_field_cost_not_context_dilution(self):
        self.fields[0]['value'] = '900.00'
        m = self.metrics()
        self.assertAlmostEqual(m['weighted_critical_field_accuracy'], 2/7)
        self.assertEqual(m['wrong_values'], 1)
        self.assertEqual(m['relationship_accuracy'], 0)
        self.assertEqual(m['all_critical_fields_correct_document_rate'], 0)

    def test_literal_and_semantic_values_are_separate(self):
        self.fields[0]['value'] = '100'
        m = self.metrics()
        self.assertEqual(m['weighted_critical_field_accuracy'], 1)
        self.assertEqual(m['literal_correct_values'], 2)
        self.assertGreater(m['field_character_error_rate'], 0)
        self.fields[2]['value'] = '123'
        self.assertEqual(self.metrics()['correct_values'], 2)

    def test_empty_outputs_remain_in_denominator(self):
        self.fields.clear()
        self.predictions['methods']['engine']['documents']['doc']['regions'] = {}
        m = self.metrics()
        self.assertEqual(m['weighted_critical_field_accuracy'], 0)
        self.assertEqual(m['missing_fields'], 3)
        self.assertEqual(m['field_character_error_rate'], 1)
        self.assertEqual(m['critical_region_character_error_rate'], 1)
        self.assertEqual(m['eligible_documents'], 1)
        self.assertIsNone(m['value_precision'])

    def test_wrong_record_cannot_match_even_when_value_is_identical(self):
        self.fields[0]['record'] = 'B'
        m = self.metrics()
        self.assertEqual((m['missing_fields'], m['extra_fields']), (1, 1))
        errors = self.run_benchmark()['methods']['engine']['documents']['doc']['extra_predictions']
        self.assertEqual(errors[0]['reason'], 'wrong_record')

    def test_row_reordering_invariant_but_value_swaps_penalized(self):
        record = deepcopy(self.references['documents']['doc']['records'][0])
        record['id'] = 'B'
        record['fields']['Amount']['value'] = '200.00'
        self.references['documents']['doc']['records'].append(record)
        more = deepcopy(self.fields)
        for f in more:
            f['record'] = 'B'
        more[0]['value'] = '200.00'
        self.fields.extend(more)
        self.fields.reverse()
        self.assertEqual(self.metrics()['value_recall'], 1)
        amounts = [f for f in self.fields if f['field'] == 'Amount']
        amounts[0]['value'], amounts[1]['value'] = amounts[1]['value'], amounts[0]['value']
        self.assertEqual(self.metrics()['wrong_values'], 2)
        self.assertEqual(self.metrics()['record_accuracy'], 0)

    def test_duplicate_predictions_penalize_precision_and_document_success(self):
        self.fields.append(deepcopy(self.fields[0]))
        m = self.metrics()
        self.assertEqual(m['value_precision'], 0.75)
        self.assertEqual(m['extra_fields'], 1)
        self.assertEqual(m['weighted_critical_field_accuracy'], 1)
        self.assertEqual(m['all_critical_fields_correct_document_rate'], 0)

    def test_best_duplicate_is_matched_once_and_bad_duplicate_remains_extra(self):
        wrong = {**self.fields[0], 'value': '900'}
        self.fields.insert(0, wrong)
        self.assertEqual(self.metrics()['correct_values'], 3)
        self.assertEqual(self.metrics()['extra_fields'], 1)

    def test_absent_field_hallucination_counts_as_extra(self):
        self.references['documents']['doc']['records'][0]['fields']['Account'] = {'status': 'absent'}
        m = self.metrics()
        self.assertEqual(m['reference_fields'], 2)
        self.assertEqual(m['extra_fields'], 1)
        self.assertEqual(m['all_critical_fields_correct_document_rate'], 0)

    def test_unreadable_fields_reported_without_false_perfect_document(self):
        self.references['documents']['doc']['records'][0]['fields']['Account'].update(status='unreadable', value=None)
        m = self.metrics()
        self.assertEqual(m['ignored_unreadable_predictions'], 1)
        self.assertEqual(m['unreadable_fields'], 1)
        self.assertEqual(m['eligible_documents'], 0)
        self.assertIsNone(m['all_critical_fields_correct_document_rate'])
        self.assertEqual(m['reference_fields'], 2)

    def test_unapplicable_required_field_is_not_a_constraint_failure(self):
        self.references['documents']['doc']['records'][0]['fields']['Account'] = {'status': 'not_applicable'}
        self.fields.pop()
        result = self.run_benchmark()['methods']['engine']['documents']['doc']
        self.assertEqual(result['constraint_violations'], [])

    def test_constraints_do_not_repair_or_override_correctness(self):
        self.profiles['profiles']['trade']['fields']['Amount']['constraints'] = {'maximum': '50'}
        result = self.run_benchmark()['methods']['engine']['documents']['doc']
        self.assertEqual(result['metrics']['value_recall'], 1)
        self.assertEqual(result['constraint_violations'][0]['reason'], 'maximum')
        self.assertEqual(result['source_constraint_violations'][0]['reason'], 'maximum')

    def test_fixed_regions_do_not_depend_on_field_detection(self):
        self.fields.clear()
        m = self.metrics()
        self.assertEqual(m['critical_region_character_error_rate'], 0)
        self.assertEqual(m['value_recall'], 0)

    def test_wrong_currency_invalidates_associated_amount_even_when_noncritical(self):
        self.profiles['profiles']['trade']['fields']['Currency']['critical'] = False
        self.fields[1]['value'] = 'EUR'
        m = self.metrics()
        self.assertEqual(m['correct_values'], 2)
        self.assertAlmostEqual(m['weighted_critical_field_accuracy'], 1/6)
        self.assertEqual(m['all_critical_fields_correct_document_rate'], 0)

    def test_duplicate_association_blocks_document_pass(self):
        self.profiles['profiles']['trade']['fields']['Currency']['critical'] = False
        self.fields.append(deepcopy(self.fields[1]))
        self.assertEqual(self.metrics()['all_critical_fields_correct_document_rate'], 0)

    def test_unreadable_noncritical_association_prevents_claiming_document_success(self):
        self.profiles['profiles']['trade']['fields']['Currency']['critical'] = False
        self.references['documents']['doc']['records'][0]['fields']['Currency'].update(status='unreadable', value=None)
        self.assertIsNone(self.metrics()['all_critical_fields_correct_document_rate'])

    def test_document_and_field_weighting_and_split_filtering(self):
        other_ref = deepcopy(self.references['documents']['doc'])
        other_ref['template_id'] = 'template-b'
        other_pred = deepcopy(self.predictions['methods']['engine']['documents']['doc'])
        extra_record = deepcopy(other_ref['records'][0])
        extra_record['id'] = 'B'
        other_ref['records'].append(extra_record)
        other_pred['fields'] = []
        self.references['documents']['other'] = other_ref
        self.predictions['methods']['engine']['documents']['other'] = other_pred
        self.assertEqual(self.run_benchmark()['ranking'][0]['critical_field_score'], 50)
        self.assertAlmostEqual(self.run_benchmark(weighting='field')['ranking'][0]['critical_field_score'], 100/3)
        other_ref['split'] = 'validation'
        self.assertEqual(self.run_benchmark()['ranking'][0]['critical_field_score'], 100)
        self.assertEqual(self.run_benchmark(split='validation')['ranking'][0]['critical_field_score'], 0)

    def test_noncritical_errors_do_not_lower_critical_score(self):
        self.profiles['profiles']['trade']['fields']['Account']['critical'] = False
        self.fields[2]['value'] = 'wrong'
        self.assertEqual(self.metrics()['weighted_critical_field_accuracy'], 1)
        self.assertEqual(self.metrics()['wrong_values'], 1)

    def test_unexpected_fields_penalize_precision_and_document_pass(self):
        self.fields.append({'record': 'A', 'field': 'Unknown', 'value': 'invented'})
        self.assertEqual(self.metrics()['value_precision'], 0.75)
        self.assertEqual(self.metrics()['all_critical_fields_correct_document_rate'], 0)

    def test_blank_and_invalid_predictions_do_not_earn_value_credit(self):
        self.fields[0]['value'] = ''
        self.fields[1]['value'] = 'not-a-code'
        m = self.metrics()
        self.assertEqual(m['missing_fields'], 1)
        self.assertEqual(m['wrong_values'], 1)
        self.assertEqual(m['correct_values'], 1)
        self.assertEqual(m['extra_fields'], 1)

    def test_literal_comparison_allows_exact_transcription_of_invalid_source_type(self):
        self.profiles['profiles']['trade']['fields']['Amount']['comparison'] = 'literal'
        self.references['documents']['doc']['records'][0]['fields']['Amount']['value'] = 'unclear-as-written'
        self.fields[0]['value'] = 'unclear-as-written'
        self.assertEqual(self.metrics()['value_recall'], 1)

    def test_bad_boxes_and_boolean_pages_rejected(self):
        for box in ([0, 0, 0, 1], [-1, 0, 1, 1], [0, 0, float('nan'), 1]):
            self.fields[0]['bbox'] = box
            with self.assertRaisesRegex(ValueError, 'bbox'):
                self.run_benchmark()
        self.fields[0]['bbox'] = list(self.box)
        self.fields[0]['page'] = True
        with self.assertRaisesRegex(ValueError, 'page'):
            self.run_benchmark()

    def test_localization_requires_page_and_box(self):
        self.fields[0].pop('bbox')
        self.fields[1]['page'] = 2
        self.assertAlmostEqual(self.metrics()['localization_recall'], 1/3)
        self.assertEqual(self.metrics()['value_recall'], 1)

    def test_verified_references_require_reviewed_profiles_and_reviewer(self):
        self.references['annotation_status'] = 'verified'
        with self.assertRaisesRegex(ValueError, 'reviewed profile'):
            self.run_benchmark()
        self.profiles['profiles']['trade'].update(review_status='reviewed', reviewed_by='Profile reviewer')
        with self.assertRaisesRegex(ValueError, 'reviewed_by'):
            self.run_benchmark()
        self.references['documents']['doc']['reviewed_by'] = 'Image reviewer'
        self.assertEqual(self.run_benchmark()['annotation_status'], 'verified')

    def test_incomplete_or_draft_references_rejected(self):
        self.references['annotation_status'] = 'draft'
        with self.assertRaisesRegex(ValueError, 'draft'):
            self.run_benchmark()
        self.references['annotation_status'] = 'synthetic'
        del self.references['documents']['doc']['records'][0]['fields']['Account']
        with self.assertRaisesRegex(ValueError, 'annotate every'):
            self.run_benchmark()

    def test_hash_mismatch_missing_document_and_unknown_region_rejected(self):
        document = self.predictions['methods']['engine']['documents']['doc']
        document['source_sha256'] = '1'*64
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.run_benchmark()
        document['source_sha256'] = '0'*64
        document['regions']['unknown'] = 'text'
        with self.assertRaisesRegex(ValueError, 'unknown reference region'):
            self.run_benchmark()
        self.predictions['methods']['engine']['documents'].clear()
        with self.assertRaisesRegex(ValueError, 'every reference document'):
            self.run_benchmark()

    def test_templates_cannot_leak_between_splits(self):
        other = deepcopy(self.references['documents']['doc'])
        other['split'] = 'train'
        self.references['documents']['other'] = other
        with self.assertRaisesRegex(ValueError, 'multiple splits'):
            self.run_benchmark()

    def test_profile_validation_catches_weights_alias_collisions_and_bad_constraints(self):
        for weight in (0, -1, True, float('nan'), float('inf')):
            profiles = deepcopy(self.profiles)
            profiles['profiles']['trade']['fields']['Amount']['weight'] = weight
            with self.assertRaisesRegex(ValueError, 'weight'):
                validate_profiles(profiles)
        self.profiles['profiles']['trade']['fields']['Account']['aliases'] = ['Amount']
        with self.assertRaisesRegex(ValueError, 'Ambiguous profile alias'):
            validate_profiles(self.profiles)

    def test_invalid_dates_fail_reference_validation_instead_of_guessing(self):
        spec = self.profiles['profiles']['trade']['fields']['Account']
        spec['value_type'] = 'date'
        self.references['documents']['doc']['records'][0]['fields']['Account']['value'] = '01/02/2026'
        with self.assertRaisesRegex(ValueError, 'date_order'):
            self.run_benchmark()
        spec['date_order'] = 'dmy'
        self.fields[2]['value'] = '2026-02-01'
        self.assertEqual(self.metrics()['value_recall'], 1)

    def test_reference_independent_extractor_uses_aliases_and_stable_table_keys(self):
        page = '| Record | Settlement Amount | Currency | Account No |\n|---|---|---|---|\n| A | 100.00 | USD | 00123 |'
        profile = self.profiles['profiles']['trade']
        result = extract_profile_fields([page], profile, 'markdown')
        self.assertEqual([(x['record'], x['field'], x['value']) for x in result['fields']],
                         [('A', 'Amount', '100.00'), ('A', 'Currency', 'USD'), ('A', 'Account', '00123')])
        aligned = {'doc': {'a': {'pages': [page]}}}
        predicted = extract_profile_predictions(aligned, self.profiles, {'doc': 'trade'}, {'a': 'family'})
        scored = evaluate_benchmark(self.profiles, self.references, predicted)
        self.assertEqual(scored['ranking'][0]['critical_field_score'], 100)
        self.assertEqual(scored['methods']['a']['metrics']['localization_recall'], 0)

    def test_extraction_does_not_invent_record_ids_for_unkeyed_tables(self):
        result = extract_profile_fields(['Amount\tCurrency\n100\tUSD'], self.profiles['profiles']['trade'])
        self.assertEqual(result['fields'], [])
        self.assertEqual(result['extraction_issues'][0]['reason'], 'table_requires_record_key')

    def test_explicit_pairs_and_adjacent_lines(self):
        result = extract_profile_fields(['Settlement Amount: 100\nAccount No:\n00123\nCurrency\nUSD'],
                                        self.profiles['profiles']['trade'])
        self.assertEqual([x['value'] for x in result['fields']], ['100', '00123', 'USD'])
        self.assertTrue(all(x['record'] == 'document' for x in result['fields']))

    def test_variants_keep_reference_metric_and_methods_fixed(self):
        bad = deepcopy(self.predictions)
        bad['methods']['engine']['documents']['doc']['fields'] = []
        result = compare_benchmark_variants(self.profiles, self.references, {'baseline': self.predictions, 'bad': bad})
        self.assertEqual(result['score_differences']['bad']['engine'], -100)
        bad['methods']['other'] = bad['methods'].pop('engine')
        with self.assertRaisesRegex(ValueError, 'same methods'):
            compare_benchmark_variants(self.profiles, self.references, {'baseline': self.predictions, 'bad': bad})

    def test_family_sensitivity_excludes_related_peers(self):
        families = {'a': 'shared', 'b': 'shared', 'c': 'independent', 'd': 'third'}
        values = [('a', 'b', 1), ('a', 'c', 0.2), ('b', 'c', 0.2), ('a', 'd', 0.6), ('b', 'd', 0.6), ('c', 'd', 0.4)]
        pairs = [{'left': a, 'right': b, 'primary_similarity': v, 'document': 'doc', 'page': 1} for a, b, v in values]
        result = consensus_family_sensitivity(pairs, families)
        self.assertAlmostEqual(result['family_balanced_scores']['a'], 40)
        self.assertAlmostEqual(result['leave_one_family_out']['third']['a'], 20)
        with self.assertRaisesRegex(ValueError, 'complete peer matrix'):
            consensus_family_sensitivity(pairs[:-1], families)

    def test_cli_and_draft_template_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in [('profiles', self.profiles), ('references', self.references), ('predictions', self.predictions)]:
                (root / f'{name}.json').write_text(json.dumps(value), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                main(['benchmark', '--profiles', str(root/'profiles.json'), '--references', str(root/'references.json'),
                      '--predictions', str(root/'predictions.json'), '--output', str(root/'results')])
            result = load_json(root/'results/results.json')
            self.assertEqual(result, self.run_benchmark())
            self.assertIn('Synthetic demonstration', (root/'results/report.md').read_text())
            (root/'source.txt').write_text('Example image placeholder')
            manifest = {'documents': {'doc': {'profile': 'trade', 'source_path': 'source.txt', 'template_id': 't'}}}
            draft = make_reference_template(self.profiles, manifest, root)
            self.assertEqual(draft['annotation_status'], 'draft')
            self.assertEqual(draft['documents']['doc']['records'][0]['fields']['Amount']['status'], 'unannotated')
            self.assertNotEqual(draft['documents']['doc']['source']['sha256'], '0'*64)

    def test_cli_template_variants_and_family_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'documents': {'doc': {'profile': 'trade', 'source_path': 'source.txt', 'template_id': 't'}}}
            pairs = [{'left': a, 'right': b, 'primary_similarity': 0.5, 'document': 'doc', 'page': 1}
                     for a, b in [('a', 'b'), ('a', 'c'), ('b', 'c')]]
            files = {'profiles': self.profiles, 'references': self.references, 'predictions': self.predictions,
                     'manifest': manifest, 'families': {'a': 'A', 'b': 'B', 'c': 'C'}, 'pairs': pairs}
            for name, value in files.items():
                (root/f'{name}.json').write_text(json.dumps(value))
            (root/'source.txt').write_text('synthetic source')
            with contextlib.redirect_stdout(io.StringIO()):
                main(['init-reference', str(root/'manifest.json'), '--profiles', str(root/'profiles.json'),
                      '--output', str(root/'draft.json')])
                main(['compare-variants', '--profiles', str(root/'profiles.json'), '--references', str(root/'references.json'),
                      '--variant', f'a={root}/predictions.json', '--variant', f'b={root}/predictions.json',
                      '--output', str(root/'comparison')])
                main(['family-sensitivity', str(root/'pairs.json'), '--families', str(root/'families.json'),
                      '--output', str(root/'sensitivity.json')])
            self.assertEqual(load_json(root/'comparison/variants.json')['score_differences']['b']['engine'], 0)
            self.assertEqual(load_json(root/'sensitivity.json')['family_balanced_scores']['a'], 50)
            original = (root/'draft.json').read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['init-reference', str(root/'manifest.json'), '--profiles', str(root/'profiles.json'),
                      '--output', str(root/'draft.json')])
            self.assertEqual((root/'draft.json').read_bytes(), original)

    def test_json_duplicate_keys_and_nonfinite_values_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.json'
            for content in ('{"a": 1, "a": 2}', '{"a": NaN}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    load_json(path)

    def test_cer_is_exact_and_can_exceed_one(self):
        self.assertEqual(edit_distance('kitten', 'sitting'), 3)
        self.assertEqual(edit_distance('', 'long'), 4)
        self.fields[0]['value'] = '9'*100
        self.assertGreater(self.metrics()['field_character_error_rate'], 1)


if __name__ == '__main__':
    unittest.main()
