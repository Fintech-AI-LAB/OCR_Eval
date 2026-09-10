import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cross_model_eval import evaluate
from evaluate_outputs import normalize_texts, compare_page, one_match
from score_page import score_page, score_models, score_page_details


class EvaluationHardeningTests(unittest.TestCase):
    def test_normalized_json_roundtrip_preserves_literal_markup_and_scores(self):
        data = {'d': {
            'a': {'pages': ['Name &lt;p&gt;'], 'input_format': 'markdown'},
            'b': {'pages': ['Name <p>'], 'input_format': 'plain'},
            'c': {'pages': ['Name'], 'input_format': 'plain'}}}
        first, pairs, _, normalized = evaluate(data)
        second, again, _, reloaded = evaluate(json.loads(json.dumps(normalized)))
        self.assertEqual(first['ranking'], second['ranking'])
        self.assertEqual(normalized, reloaded)
        self.assertEqual(pairs, again)
        self.assertEqual(normalized['d']['a']['input_format'], 'plain')

    def test_inline_tags_preserve_words_but_blocks_and_cells_separate(self):
        for text in ['inter<strong>national</strong>', '<span>inter</span>national',
                     'inter<em><b>national</b></em>']:
            self.assertEqual(normalize_texts([text]), ['international'])
        self.assertEqual(normalize_texts(['<p>one</p><p>two</p>', '<td>one</td><td>two</td>']),
                         ['one two', 'one two'])

    def test_accounting_parentheses_preserved_without_inferred_sign(self):
        row = compare_page('d', 1, 'a', 'b', 'amount (100.00)', 'amount 100.00')
        self.assertLess(row['token_overlap'], 1)
        self.assertEqual(row['numeric_agreement'], 0)
        self.assertEqual(row['only_left_numeric_strings'], ['(100.00)'])
        row = compare_page('d', 1, 'a', 'b', 'amount ( 100.00 )', 'amount (100.00)')
        self.assertEqual(row['token_overlap'], 1)

    def test_numeric_diagnostics_count_missing_occurrences(self):
        row = compare_page('d', 1, 'a', 'b', '100 100 200', '100 200')
        self.assertEqual(row['numeric_agreement'], 0.8)
        self.assertEqual(row['unmatched_left_numeric_counts'], {'100': 1})
        self.assertEqual(row['unmatched_right_numeric_counts'], {})

    def test_ambiguous_or_missing_artifacts_fail_instead_of_selecting_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaises(FileNotFoundError): one_match(base.glob('*.json'), 'test')
            (base / 'run1.json').write_text('{}')
            self.assertEqual(one_match(base.glob('*.json'), 'test'), base / 'run1.json')
            (base / 'run2.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                one_match(base.glob('*.json'), 'test')

    def test_fast_scores_match_diagnostics_without_computing_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f'{i}.txt' for i in range(3)]
            for path, text in zip(paths, ['alpha beta (100)', 'beta alpha 100', 'alpha']):
                path.write_text(text)
            details = score_page_details(paths)
            with patch('score_page.compare_page', side_effect=AssertionError('Unnecessary alignment')):
                self.assertEqual(score_page(paths), details['score'])
                self.assertEqual(score_models(paths), details['model_scores'])
            self.assertEqual(score_page(paths, 0.5), score_page_details(paths, 0.5)['score'])


if __name__ == '__main__': unittest.main()
