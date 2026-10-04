import contextlib
import io
import itertools
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'standalone_ocr_eval'))
from ontology_regions import extract_ontology_regions, region_metrics, _assignment
from cross_model_eval import evaluate
from score_page import score_models, score_page_details
from evaluate_outputs import compare_page
from rank_ocr import calculate, ENGINES
import ocr_eval
from ocr_eval.cli import main


class OntologyRegionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ontology = self.root / 'ontology.json'
        self.ontology.write_text(json.dumps({'concepts': {
            'Amount': {}, 'AccountNumber': {'aliases': ['Account No']},
            'SettlementDate': {}, 'Invoice': {'fields': ['Reference', 'To']},
            'Payment': {'fields': ['Reference', 'Agent', 'Basis']},
            'Account': {'fields': ['Agent', 'Basis']}}}))

    def regions(self, text, context=16, fmt='plain'):
        return extract_ontology_regions(text, self.ontology, fmt, context)

    def compare(self, a, b, context=16):
        return region_metrics(self.regions(a, context), self.regions(b, context))

    def files(self, *texts):
        paths = []
        for index, text in enumerate(texts):
            path = self.root / f'{index}.md'
            path.write_text(text)
            paths.append(path)
        return paths

    def data(self, *texts, fmt='plain'):
        return {'doc': {str(i): {'pages': [t], 'input_format': fmt} for i, t in enumerate(texts)}}

    def test_aliases_and_camel_case_localize_same_passage(self):
        self.assertEqual(self.compare('AccountNumber: 00123', 'Account No: 00123')['ontology_region_agreement'], 1)
        self.assertEqual(self.compare('SettlementDate: 2026-01-02', 'Settlement Date: 2026-01-02')['ontology_region_agreement'], 1)
        region = self.regions('line one\nAmount: 100\nline three')['regions'][0]
        self.assertEqual((region['start_line'], region['end_line']), (1, 3))
        self.assertIn('100', region['text'])

    def test_content_and_local_order_are_evaluated(self):
        a = 'Amount: 100\nAccountNumber: 001'
        swapped = 'Amount: 001\nAccountNumber: 100'
        result = self.compare(a, swapped)
        self.assertEqual(result['region_matches'][0]['token_overlap'], 1)
        self.assertLess(result['ontology_region_agreement'], 1)
        self.assertEqual(result['ontology_anchor_agreement'], 1)
        for value in ('-100', '100%', '0100', '(100)', '100.00'):
            self.assertLess(self.compare('Amount: 100', 'Amount: ' + value)['ontology_region_agreement'], 1)

    def test_ambiguous_owners_and_invalid_dates_are_not_penalties(self):
        text = 'Reference: ABC\nSettlement Date: unreadable'
        result = self.compare(text, text)
        self.assertEqual(result['ontology_region_agreement'], 1)
        self.assertGreater(result['left_region_localization']['ambiguous_anchor_count'], 0)
        self.assertTrue(any(len(a['candidate_fields']) > 1 for a in self.regions(text)['anchors']))

    def test_outside_text_ignored_and_overlap_counted_once(self):
        a = 'alpha ' * 60 + 'Amount: 100 ' + 'tail ' * 60
        b = 'beta ' * 60 + 'Amount: 100 ' + 'tail ' * 60
        result = self.compare(a, b, context=2)
        # Nearby words differ; changes outside the region do not matter.
        self.assertLess(result['ontology_region_agreement'], 1)
        b = 'beta ' * 58 + 'alpha alpha Amount: 100 ' + 'tail ' * 60
        self.assertEqual(self.compare(a, b, context=2)['ontology_region_agreement'], 1)
        regions = self.regions('Amount 100 Account Number 001 Amount 200', context=3)
        budget = {}
        for r in regions['regions']:
            for offset, weight in zip(range(r['start_token'], r['end_token']), r['token_weights']):
                budget[offset] = budget.get(offset, 0) + weight
        self.assertTrue(all(abs(weight - 1) < 1e-12 for weight in budget.values()))
        self.assertEqual(len(budget), regions['localization']['selected_tokens'])
        self.assertLessEqual(regions['localization']['selected_token_fraction'], 1)
        self.assertLess(result['left_region_localization']['selected_token_fraction'], 0.1)

    def test_common_prose_word_does_not_select_region(self):
        result = self.regions('Send this to somebody. To whom it concerns.')
        self.assertFalse(result['regions'])
        self.assertEqual(len(result['suppressed_anchors']), 2)

    def test_missing_and_repeated_regions_are_not_reused(self):
        padding = 'gap ' * 30
        a = padding + 'Amount 100 ' + padding + 'Amount 200 ' + padding
        b = padding + 'Amount 200 ' + padding + 'Amount 100 ' + padding
        self.assertEqual(self.compare(a, b, context=1)['ontology_region_agreement'], 1)
        result = self.compare(a, 'Amount 100', context=1)
        self.assertLess(result['ontology_region_agreement'], 1)
        self.assertEqual(len(result['region_matches']), 1)
        self.assertEqual(len(result['unmatched_left_ontology_regions']), 1)
        self.assertEqual(self.compare(a, '', context=1)['ontology_region_agreement'], 0)
        self.assertIsNone(self.compare('ordinary text', '')['ontology_region_agreement'])

    def test_table_region_compares_values_without_column_parsing(self):
        a = '| Amount | AccountNumber |\n|---|---|\n| 100 | 001 |'
        b = '<table><tr><th>Amount</th><th>AccountNumber</th></tr><tr><td>100</td><td>001</td></tr></table>'
        self.assertEqual(region_metrics(self.regions(a, fmt='markdown'), self.regions(b, fmt='html'))['ontology_region_agreement'], 1)
        self.assertLess(region_metrics(self.regions(a, fmt='markdown'), self.regions(a.replace('100', '900'), fmt='markdown'))['ontology_region_agreement'], 1)

    def test_long_unions_keep_each_token_once(self):
        text = ' '.join('Amount ' + 'word ' * 12 for _ in range(35))
        regions = self.regions(text)
        token_budget, bigram_budget = {}, {}
        for r in regions['regions']:
            for offset, weight in zip(range(r['start_token'], r['end_token']), r['token_weights']):
                token_budget.setdefault(offset, []).append(weight)
            for offset, weight in zip(range(r['start_token'], r['end_token'] - 1), r['bigram_weights']):
                bigram_budget.setdefault(offset, []).append(weight)
        self.assertEqual(sorted(token_budget), list(range(regions['localization']['total_tokens'])))
        for budget in (token_budget, bigram_budget):
            self.assertTrue(all(abs(math.fsum(weights) - 1) < 1e-12 for weights in budget.values()))
        self.assertAlmostEqual(sum(r['token_mass'] for r in regions['regions']), regions['localization']['selected_tokens'])
        self.assertTrue(all(r['anchors'] for r in regions['regions']))
        self.assertGreater(len(regions['regions']), 1)
        self.assertEqual(self.compare(text, text)['ontology_region_agreement'], 1)

    def test_generic_labels_do_not_match_unrelated_sections(self):
        result = self.compare('Bank endorsement Agent authorised signature Basis collection',
                              'Invoice company Agent shipping address Basis sale')
        self.assertEqual(result['ontology_region_agreement'], 0)
        self.assertFalse(result['region_matches'])
        self.assertEqual(result['context_rejected_candidate_count'], 2)
        # Ambiguity is still a diagnostic, not an automatic score penalty.
        same = 'Bank endorsement Agent authorised signature Basis collection'
        self.assertEqual(self.compare(same, same)['ontology_region_agreement'], 1)

    def test_generic_label_with_context_still_scores_changed_content(self):
        result = self.compare('Invoice supplier Reference ABC payment terms net thirty days',
                              'Invoice supplier Reference XYZ payment terms net thirty days')
        self.assertGreater(result['ontology_region_agreement'], 0)
        self.assertLess(result['ontology_region_agreement'], 1)
        reference = next(r for r in result['region_matches'] if r['central_anchor_tag'] == 'label:reference')
        self.assertTrue(reference['context_evidence']['required'])
        self.assertTrue(reference['context_evidence']['eligible'])

    def test_local_insertion_does_not_shift_distant_region_boundaries(self):
        parts = [f'Amount item{i} ' + 'detail ' * 12 for i in range(40)]
        changed = list(parts)
        changed[1] += 'page 2 '
        result = self.compare(' '.join(parts), ' '.join(changed))
        distant = [r for r in result['region_matches'] if r['left_region'] >= 6]
        self.assertEqual(len(distant), 35)
        self.assertTrue(all(abs(r['similarity'] - 1) < 1e-12 for r in distant))
        self.assertGreater(result['ontology_region_agreement'], 0.99)

    def test_central_anchor_must_match_even_when_neighbor_tags_overlap(self):
        result = self.compare('Amount 100 Account Number 001', 'Account Number 001')
        self.assertTrue(all(r['central_anchor_tag'] == 'concept:accountnumber' for r in result['region_matches']))
        self.assertEqual(len(result['unmatched_left_ontology_regions']), 1)

    def test_assignment_matches_exhaustive_optimum(self):
        rng = random.Random(14)
        for rows, columns in ((2, 3), (3, 2), (3, 3), (1, 4)):
            for _ in range(12):
                weights = [[rng.randrange(8) for _ in range(columns)] for _ in range(rows)]
                size = max(rows, columns)
                padded = [row + [0] * (size - columns) for row in weights] + [[0] * size for _ in range(size - rows)]
                optimum = max(sum(padded[i][j] for i, j in enumerate(p)) for p in itertools.permutations(range(size)))
                self.assertEqual(sum(weights[i][j] for i, j in _assignment(weights)), optimum)
        self.assertEqual(_assignment([]), [])
        self.assertEqual(_assignment([[], []]), [])

    def test_default_api_cli_and_standalone_agree_without_value_extraction(self):
        texts = ['Amount: 100\nAccountNumber: 001', 'Amount: 900\nAccountNumber: 001', 'Amount: 100\nAccountNumber: 001']
        paths = self.files(*texts)
        with patch('score_page.extract_field_values', side_effect=AssertionError('Exact extraction')), \
             patch('cross_model_eval.extract_field_values', side_effect=AssertionError('Exact extraction')), \
             patch('evaluate_outputs.extract_field_values', side_effect=AssertionError('Exact extraction')):
            details = score_page_details(paths, ontology_path=self.ontology)
            result, pairs, volumes, normalized = evaluate(self.data(*texts), ontology_path=self.ontology)
            self.assertEqual(compare_page('doc', 1, '0', '1', *texts[:2], self.ontology)['ontology_region_agreement'], pairs[0]['ontology_region_agreement'])
        scores = {r['model']: r['consensus_score'] for r in result['ranking']}
        self.assertEqual(details['model_scores'], [scores[str(i)] for i in range(3)])
        self.assertEqual(ocr_eval.score_page_details(paths, ontology_path=self.ontology), details)
        self.assertEqual(ocr_eval.evaluate(self.data(*texts), ontology_path=self.ontology), (result, pairs, volumes, normalized))
        self.assertEqual(result['settings']['primary_metric'], 'ontology_region_agreement')
        self.assertIsNone(pairs[0]['ontology_field_value_agreement'])
        self.assertEqual(evaluate(normalized, ontology_path=self.ontology)[0], result)
        with patch('score_page.compare_page', side_effect=AssertionError('Unnecessary global alignment')):
            self.assertEqual(score_models(paths, ontology_path=self.ontology), details['model_scores'])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(['page', *map(str, paths), '--ontology', str(self.ontology), '--region-context-tokens', '2'])
        self.assertEqual(json.loads(output.getvalue())['settings']['region_policy']['context_tokens_each_side'], 2)

    def test_no_anchors_excluded_and_no_ontology_uses_all_tokens(self):
        data = self.data('Amount 100', '', '')
        for record in data['doc'].values():
            record['pages'].append('ordinary prose')
        result, pairs, _, _ = evaluate(data, ontology_path=self.ontology)
        self.assertEqual(len(result['excluded_pages']), 1)
        self.assertEqual(len(result['localization']), 6)
        self.assertTrue(all(p['primary_similarity'] == 0 for p in pairs))
        with self.assertRaisesRegex(ValueError, 'No ontology regions located'):
            score_page_details(self.files('ordinary prose', 'ordinary prose'), ontology_path=self.ontology)
        baseline = score_page_details(self.files('ordinary prose', 'ordinary prose'), ontology_path=None)
        self.assertEqual(baseline['score'], 100)
        self.assertEqual(baseline['settings']['primary_metric'], 'token_overlap')
        self.assertFalse(baseline['localization'])

    def test_region_matching_symmetric_and_context_validated(self):
        a, b = 'Amount 100 ' + 'gap ' * 40 + 'AccountNumber 001', 'Account Number 001'
        self.assertEqual(self.compare(a, b, context=2)['ontology_region_agreement'], self.compare(b, a, context=2)['ontology_region_agreement'])
        for invalid in (0, -1, 129, True, 1.5, None):
            with self.assertRaisesRegex(ValueError, 'region_context_tokens'):
                self.regions(a, context=invalid)
            with self.assertRaisesRegex(ValueError, 'region_context_tokens'):
                evaluate(self.data(a, a, a), ontology_path=self.ontology, region_context_tokens=invalid)

    def test_saved_regions_can_be_ranked_and_old_version_rejected(self):
        texts = ['Amount 100', 'Amount 100', 'Amount 900', 'Amount 900']
        rows = [compare_page('doc', 1, a, b, texts[i], texts[j], self.ontology)
                for i, a in enumerate(ENGINES) for j, b in enumerate(ENGINES) if i < j]
        self.assertAlmostEqual(calculate(rows)['doc'][ENGINES[0]][0], 0.5)
        rows[0]['algorithm_version'] = '6.0'
        with self.assertRaisesRegex(ValueError, 'current ontology region'):
            calculate(rows)


if __name__ == '__main__':
    unittest.main()
