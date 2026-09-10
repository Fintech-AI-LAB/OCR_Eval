import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cross_model_eval import evaluate, validate

class EvaluationTests(unittest.TestCase):
    def data(self, *texts):
        return {'doc': {name: {'pages': [text]} for name, text in zip('abcd', texts)}}

    def test_identical_outputs_tie(self):
        result, _, _, _ = evaluate(self.data('same text', 'same text', 'same text'))
        self.assertEqual([r['rank'] for r in result['ranking']], [1, 1, 1])
        self.assertTrue(all(r['consensus_score'] == 100 for r in result['ranking']))

    def test_no_reward_for_shared_empty_output(self):
        result, pairs, _, _ = evaluate(self.data('invoice 00123', '', ''))
        self.assertTrue(all(p['combined_similarity'] == 0 for p in pairs))
        with self.assertRaises(ValueError): evaluate(self.data('', '', ''))

    def test_page_alignment_is_required(self):
        data = self.data('a', 'a', 'a'); data['doc']['b']['pages'].append('b')
        with self.assertRaises(ValueError): validate(data)

    def test_weighting_changes_document_influence(self):
        data = self.data('one two', 'one two', 'different')
        data['long'] = {'a': {'pages': ['different'] * 4},
                        'b': {'pages': ['one two'] * 4}, 'c': {'pages': ['one two'] * 4}}
        doc = evaluate(data, weighting='document')[0]['ranking']
        page = evaluate(data, weighting='page')[0]['ranking']
        self.assertNotEqual(doc, page)

    def test_formatting_not_scored_and_digits_preserved(self):
        result, pairs, _, normalized = evaluate(self.data('# ID\n00123', '<p>ID 00123</p>', 'ID 123'))
        self.assertEqual(normalized['doc']['a']['pages'], ['ID 00123'])
        self.assertEqual(pairs[0]['token_overlap'], 1)
        self.assertEqual(pairs[1]['only_left_numeric_strings'], ['00123'])
        self.assertEqual(pairs[1]['only_right_numeric_strings'], ['123'])

    def test_invalid_weight(self):
        for w in [-1, 2, float('nan')]:
            with self.assertRaises(ValueError): evaluate(self.data('a','b','c'), w)

    def test_primary_ranking_ignores_order_but_keeps_diagnostics(self):
        result, pairs, _, _ = evaluate(self.data('a b c', 'c a b', 'b c a'))
        self.assertEqual([r['consensus_score'] for r in result['ranking']], [100, 100, 100])
        self.assertEqual(result['settings']['primary_metric'], 'token_count_overlap')
        self.assertTrue(all(p['primary_similarity'] == 1 for p in pairs))
        self.assertTrue(any(p['sequence_similarity'] < 1 for p in pairs))
        self.assertTrue(all(r['mean_numeric_agreement'] is None for r in result['pairwise_summary']))

if __name__ == '__main__': unittest.main()
