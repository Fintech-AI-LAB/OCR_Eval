from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone_ocr_eval'))
from ocr_eval import evaluate_saved
from ocr_eval.cli import main


def bundle():
    return {'data': {'doc': {m: {'pages': ['Amount 900' if m in 'ab' else 'Amount 100']}
                             for m in 'abcdef'}},
            'families': {m: [m] for m in 'abcdef'}, 'panel': list('abcdef')}


class SavedPackageTests(unittest.TestCase):
    def test_threshold_changes_passages_but_not_full_text_scores(self):
        source = bundle()
        before = deepcopy(source)
        low = evaluate_saved(**source)
        high = evaluate_saved(**source, min_families=3, min_fraction=0.5)
        self.assertEqual(source, before)
        self.assertEqual(low['document_agreement'], high['document_agreement'])
        self.assertEqual(low['passage_corroboration']['required_family_count'], 2)
        row = next(r for r in low['passage_corroboration']['passages'] if r['method'] == 'a')
        self.assertEqual(row['status'], 'corroborated')
        row = next(r for r in high['passage_corroboration']['passages'] if r['method'] == 'a')
        self.assertEqual(row['status'], 'uncertain')

    def test_document_only_candidate_receives_score_but_no_passage_counts(self):
        source = bundle()
        source['families']['unaligned'] = ['unknown']
        source['data']['doc']['unaligned'] = {'pages': None, 'document_text': 'Amount 900'}
        result = evaluate_saved(**source)
        row = next(r for r in result['ranking'] if r['method'] == 'unaligned')
        self.assertIsInstance(row['score'], float)
        self.assertIsNone(row['corroborated_passages'])
        self.assertIsNone(row['page_agreement'])
        source['panel'].append('unaligned')
        with self.assertRaisesRegex(ValueError, 'alignment'):
            evaluate_saved(**source)

    def test_cli_writes_reviewable_outputs_with_two_family_default(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.json'
            output = Path(directory) / 'results'
            source.write_text(json.dumps(bundle()), encoding='utf-8')
            main(['evaluate-saved', str(source), '--output', str(output)])
            result = json.loads((output / 'results.json').read_text())
            self.assertEqual(result['passage_corroboration']['required_family_count'], 2)
            self.assertIn('Score /100', (output / 'report.md').read_text())
            for name in ('comparison.csv', 'document_agreement.json', 'page_agreement.json',
                         'passage_corroboration.json', 'passage_corroboration.csv',
                         'uncertain_passages.csv', 'flagged_passages.csv'):
                self.assertTrue((output / name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
