"""Guard the raw-text contract and reject incomplete/stale experiment inputs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_paddleocr_v4 import response_from_result
from evaluate_paddleocr_v4 import verify_run
from ocr_common import destination_for, write_json


class PaddleExperimentTests(unittest.TestCase):
    def response(self, page=1):
        # Markup-like OCR must remain literal; wrong digits must not be repaired.
        raw = {'rec_texts': ['<Account> 00123 | -5.00', 'Wrong digit §9'],
               'rec_scores': [0.9, 0.7], 'rec_polys': [[[0, 0]] * 4] * 2}
        return response_from_result(SimpleNamespace(json={'res': raw}), page, (800, 1000), 1.2)

    def test_raw_ocr_is_lossless_and_plain(self):
        r = self.response()
        self.assertEqual(r['text'], '<Account> 00123 | -5.00\nWrong digit §9')
        self.assertEqual(r['input_format'], 'plain')
        self.assertEqual(r['raw']['rec_texts'][0], r['text'].splitlines()[0])

    def test_reject_mismatched_output_arrays(self):
        r = self.response()
        r['raw']['rec_scores'].pop()
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            response_from_result(SimpleNamespace(json={'res': r['raw']}), 1, (800, 1000), 1)

    def fixture(self, root):
        source = root / 'source.pdf'
        source.write_bytes(b'synthetic source bytes; no PDF inference in this test')
        output = root / 'output'
        config = {'backend': 'paddleocr-v4', 'dpi': 144}
        folder = destination_for(source, root, output, config)
        write_json(folder / 'source.json', {'source': source.name, **config, 'responses': 2})
        for page in (1, 2):
            write_json(folder / f'response-{page:04d}.json', self.response(page))
        record = {'source': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'run': str(folder.relative_to(output)), 'pages': 2,
                  'artifact_sha256': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in folder.glob('*.json')}}
        return source, output, folder, record

    def test_reject_missing_page(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output, folder, record = self.fixture(Path(temp))
            verify_run(record, source, output, 2)
            (folder / 'response-0002.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                verify_run(record, source, output, 2)

    def test_reject_changed_source_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output, folder, record = self.fixture(Path(temp))
            response = folder / 'response-0001.json'
            data = json.loads(response.read_text())
            data['text'] = 'manual correction'
            write_json(response, data)
            with self.assertRaisesRegex(ValueError, 'Artifact hash mismatch'):
                verify_run(record, source, output, 2)
            source.write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
                verify_run(record, source, output, 2)

    def test_reject_reordered_pages_even_with_new_artifact_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output, folder, record = self.fixture(Path(temp))
            response = folder / 'response-0001.json'
            write_json(response, self.response(page=2))
            record['artifact_sha256'][str(response.relative_to(output))] = hashlib.sha256(response.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'Page order mismatch'):
                verify_run(record, source, output, 2)


if __name__ == '__main__':
    unittest.main()
