import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.score_page import score_page, score_models


class PageScoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def files(self, texts, suffix='.txt'):
        paths = []
        for index, text in enumerate(texts):
            path = Path(self.tmp.name) / f'{index}{suffix}'
            path.write_text(text, encoding='utf-8')
            paths.append(path)
        return paths

    def test_known_scores_and_order(self):
        paths = self.files(['alpha beta', 'alpha beta', 'gamma delta'])
        self.assertEqual(score_models(paths), [50, 50, 0])
        self.assertAlmostEqual(score_page(paths), 100 / 3)
        self.assertEqual(score_models(paths[::-1]), [0, 50, 50])

    def test_normalization_and_two_outputs(self):
        self.assertEqual(score_page(self.files(['# Hello **world**', '<p>hello world</p>'])), 100)

    def test_supported_json_formats(self):
        values = [{'pages': [{'markdown': 'hello world'}]},
                  [{'content': 'hello world'}],
                  {'full_page_text': [{'lines': [{'text': 'hello world'}]}]}]
        self.assertEqual(score_page(self.files([json.dumps(v) for v in values], '.json')), 100)

    def test_reject_multi_page(self):
        paths = self.files([json.dumps({'pages': [{'markdown': 'a'}, {'markdown': 'b'}]}),
                            json.dumps({'text': 'a'})], '.json')
        with self.assertRaisesRegex(ValueError, 'exactly one page'):
            score_page(paths)

    def test_empty_and_duplicate_inputs(self):
        paths = self.files(['', '!!!', 'word'])
        self.assertEqual(score_page(paths), 0)
        with self.assertRaises(ValueError): score_page(paths[:2])
        with self.assertRaises(ValueError): score_page([paths[2], paths[2]])
        with self.assertRaises(ValueError): score_page(paths, float('nan'))


if __name__ == '__main__':
    unittest.main()
