import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone_ocr_eval'))
from ocr_eval import evaluate, score_page, score_page_details
from ocr_eval.cli import main


class StandaloneLibraryTests(unittest.TestCase):
    def test_default_and_replaceable_ontology(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'a.txt', root / 'b.txt'
            a.write_text('IBAN hello', encoding='utf-8')
            b.write_text('typo hello', encoding='utf-8')
            self.assertEqual(score_page([a, b]), 50)
            ontology = root / 'terms.json'
            ontology.write_text(json.dumps({'concepts': {'IBAN': {'fields': []}}}), encoding='utf-8')
            self.assertAlmostEqual(score_page([a, b], ontology_path=ontology, ontology_mode='mentions'), 100 / 3)
            detail = score_page_details([a, b])
            self.assertIsNone(detail['settings']['ontology'])
            self.assertEqual(detail['settings']['primary_metric'], 'token_overlap')
            self.assertEqual(detail['score'], 50)
            with self.assertRaises(FileNotFoundError):
                score_page([a, b], ontology_path=root / 'missing.json')

    def test_batch_and_cli_without_repository_files(self):
        data = {'doc': {name: {'pages': [text], 'input_format': 'plain'} for name, text in
                        [('a', 'IBAN hello'), ('b', 'typo hello'), ('c', 'IBAN hello')]}}
        result, _, _, _ = evaluate(data)
        self.assertIsNone(result['settings']['ontology'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.json'
            source.write_text(json.dumps(data), encoding='utf-8')
            output = root / 'results'
            with contextlib.redirect_stdout(io.StringIO()):
                main(['batch', str(source), '--output', str(output)])
            self.assertEqual(json.loads((output / 'results.json').read_text())['ranking'], result['ranking'])
            self.assertTrue((output / 'ranking.csv').is_file())

    def test_fable_json_keeps_literal_markup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'fable.json'
            peer = root / 'peer.txt'
            source.write_text(json.dumps({'full_page_text': [{'lines': [{'text': 'Name <p> literal'}]}]}),
                              encoding='utf-8')
            peer.write_text('Name <p> literal', encoding='utf-8')
            self.assertEqual(score_page([source, peer]), 100)


if __name__ == '__main__':
    unittest.main()
