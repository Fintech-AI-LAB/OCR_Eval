import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_outputs import clean, normalize_texts, compare_page
from cross_model_eval import evaluate
from score_page import score_models


class ScoringRegressionTests(unittest.TestCase):
    def test_meaningful_numbers_change_score(self):
        for a, b in [('USD -100.00', 'USD 100.00'), ('rate 5%', 'rate 5'),
                     ('amount 1.234', 'amount 1,234'), ('ID 00123', 'ID 123')]:
            row = compare_page('x', 1, 'a', 'b', a, b)
            self.assertLess(row['token_overlap'], 1)
            self.assertLess(row['sequence_similarity'], 1)
            self.assertEqual(row['numeric_agreement'], 0)
            self.assertTrue(row['only_left_numeric_strings'])

    def test_plain_text_is_not_html(self):
        text = 'Name <REDACTED> <p>literal</p> &lt;value&gt;'
        self.assertEqual(normalize_texts([text], ['plain']), [text])

    def test_extracted_literals_survive_normalization(self):
        text = normalize_texts(['<p>Name &lt;p&gt; <REDACTED></p>'])[0]
        self.assertEqual(text, 'Name <p> <REDACTED>')
        self.assertEqual(clean(text), clean(clean(text)))
        self.assertEqual(normalize_texts([text], ['plain']), [text])

    def test_corroborated_hyphenation(self):
        self.assertEqual(normalize_texts(['inter-\nnational', 'international']),
                         ['international', 'international'])
        self.assertEqual(normalize_texts(['well-\nknown', 'well-known']),
                         ['well- known', 'well-known'])
        self.assertEqual(clean('inter\u00adnational'), 'international')

    def test_empty_policy_matches_between_apis(self):
        texts = ['!!!', '???', 'word']
        data = {'page': {str(i): {'pages': [t], 'input_format': 'plain'} for i, t in enumerate(texts)}}
        batch = evaluate(data)[0]['ranking']
        self.assertEqual([r['consensus_score'] for r in batch], [0, 0, 0])
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f'{i}.txt' for i in range(3)]
            for path, text in zip(paths, texts): path.write_text(text)
            self.assertEqual(score_models(paths), [0, 0, 0])
            paths[-1].write_text('...')
            with self.assertRaises(ValueError): score_models(paths)
        data['page']['2']['pages'] = ['...']
        with self.assertRaises(ValueError): evaluate(data)

    def test_numeric_agreement_without_numbers_is_unavailable(self):
        self.assertIsNone(compare_page('x', 1, 'a', 'b', 'hello', 'hello')['numeric_agreement'])

    def test_mixed_files_match_batch_and_preserve_json_literals(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ('a.json', 'b.md', 'c.txt')]
            literal = 'Name <p> -100.00'
            paths[0].write_text(json.dumps({'pages': [{'text': literal}]}))
            paths[1].write_text('Name &lt;p&gt; 100.00')
            paths[2].write_text(literal)
            scores = score_models(paths)
            data = {'page': {
                'a': {'pages': [literal], 'input_format': 'plain'},
                'b': {'pages': ['Name &lt;p&gt; 100.00'], 'input_format': 'markdown'},
                'c': {'pages': [literal], 'input_format': 'plain'}}}
            batch = {r['model']: r['consensus_score'] for r in evaluate(data)[0]['ranking']}
            for name, score in zip('abc', scores):
                self.assertAlmostEqual(score, batch[name])
            self.assertGreater(scores[0], scores[1])


if __name__ == '__main__':
    unittest.main()
