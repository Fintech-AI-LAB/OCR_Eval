import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ontology_metrics import load_ontology, ontology_metrics, entity_counts
from evaluate_outputs import tokens, counter_overlap
from collections import Counter
from cross_model_eval import evaluate
from score_page import score_page, score_page_details


class OntologyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'ontology.json'
        self.path.write_text(json.dumps({'concepts': {
            'Account': {'fields': ['AccountNumber'], 'children': []},
            'SettlementDate': {}, 'Date': {}, 'IBAN': {}}}))

    def metric(self, a, b, weight=3):
        return ontology_metrics(tokens(a), tokens(b), self.path, weight)

    def test_entity_error_costs_more_than_ordinary_error(self):
        important = self.metric('IBAN hello', 'typo hello')
        ordinary = self.metric('IBAN hello', 'IBAN typo')
        self.assertAlmostEqual(important['ontology_weighted_overlap'], 1 / 3)
        self.assertAlmostEqual(ordinary['ontology_weighted_overlap'], 3 / 4)
        self.assertEqual(important['unmatched_left_ontology_entities'], {'iban': 1})

    def test_phrase_boundaries_longest_match_and_no_double_count(self):
        trie, _ = load_ontology(self.path)
        self.assertEqual(entity_counts(tokens('Account Number accountancy Settlement Date IBAN IBAN'), trie),
                         {'account number': 1, 'settlement date': 1, 'iban': 2})
        self.assertEqual(entity_counts(tokens('SettlementDate'), trie), {'settlementdate': 1})
        self.assertEqual(self.metric('Settlement Date', 'Settlement Date')['ontology_weighted_total'], 12)

    def test_symmetry_repetitions_and_empty_policy(self):
        a, b = 'IBAN IBAN hello', 'hello IBAN'
        self.assertEqual(self.metric(a, b)['ontology_weighted_overlap'],
                         self.metric(b, a)['ontology_weighted_overlap'])
        self.assertLess(self.metric(a, b)['ontology_entity_agreement'], 1)
        self.assertEqual(self.metric('', '')['ontology_weighted_overlap'], 0)
        self.assertIsNone(self.metric('hello', 'world')['ontology_entity_agreement'])
        self.assertEqual(self.metric('IBAN hello', 'hello IBAN')['ontology_weighted_overlap'], 1)

    def test_unit_weight_recovers_original_and_invalid_weights_fail(self):
        for a, b in [('IBAN hello', 'typo hello'), ('Settlement Date', 'Date Settlement'), ('', '')]:
            self.assertEqual(self.metric(a, b, 1)['ontology_weighted_overlap'],
                             counter_overlap(Counter(tokens(a)), Counter(tokens(b))))
        for weight in (0, 0.5, float('inf'), float('nan')):
            with self.assertRaises(ValueError): self.metric('a', 'b', weight)

    def test_all_entry_points_use_weight_and_record_provenance(self):
        texts = ['IBAN hello', 'IBAN typo', 'typo hello']
        data = {'doc': {str(i): {'pages': [t]} for i, t in enumerate(texts)}}
        result, pairs, _, _ = evaluate(data, ontology_path=self.path)
        paths = []
        for i, text in enumerate(texts):
            path = Path(self.tmp.name) / f'{i}.txt'
            path.write_text(text)
            paths.append(path)
        detail = score_page_details(paths, ontology_path=self.path)
        self.assertAlmostEqual(detail['score'], score_page(paths, ontology_path=self.path))
        batch = {r['model']: r['consensus_score'] for r in result['ranking']}
        self.assertEqual(detail['model_scores'], [batch[str(i)] for i in range(3)])
        self.assertEqual(len(result['settings']['ontology']['sha256']), 64)
        self.assertGreater(batch['1'], batch['2'])
        self.assertNotEqual(pairs[0]['token_overlap'], pairs[0]['primary_similarity'])

    def test_document_metadata_is_not_a_term_source(self):
        self.path.write_text(json.dumps({'concepts': {'IBAN': {'definition': 'ignore previous instructions'}},
                                         'metadata': {'name': 'hello'}}))
        self.assertIsNone(self.metric('hello previous', 'hello')['ontology_entity_agreement'])

    def test_invalid_or_missing_ontology_fails(self):
        for value in ({}, {'concepts': []}, {'concepts': {'IBAN': {'fields': 'bad'}}}):
            self.path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): load_ontology(self.path)
        with self.assertRaises(FileNotFoundError): load_ontology(self.path.with_name('missing.json'))


if __name__ == '__main__':
    unittest.main()
