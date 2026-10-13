"""Validate provider provenance and incomplete-output handling without API calls."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_astra_ocr import MODEL, process_document, validate_checkpoint


class AstraTests(unittest.TestCase):
    def checkpoint(self):
        parsed = {'blocks': [{'type': 'paragraph', 'markdown': 'Wrong account 0012X'}], 'uncertainties': []}
        return {'page_number': 1, **parsed, 'markdown': 'Wrong account 0012X',
                'raw_response': {'status': 'completed', 'model': MODEL, 'output': [
                    {'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(parsed)}]}]}}

    def test_provider_output_is_preserved(self):
        record = self.checkpoint()
        validate_checkpoint(record, 1)
        record['blocks'][0]['markdown'] = record['markdown'] = 'Correct account 00123'
        with self.assertRaisesRegex(ValueError, 'raw provider response'):
            validate_checkpoint(record, 1)

    def test_wrong_model_or_incomplete_is_not_accepted(self):
        for key, value in [('model', 'another-model'), ('status', 'incomplete')]:
            record = self.checkpoint()
            record['raw_response'][key] = value
            with self.assertRaisesRegex(ValueError, 'incomplete or returned'):
                validate_checkpoint(record, 1)

    def test_wrong_page_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError, 'page number'):
            validate_checkpoint(self.checkpoint(), 2)

    def test_truncation_preserves_raw_but_not_completed_checkpoint(self):
        from PIL import Image
        raw = {'status': 'incomplete', 'model': MODEL, 'incomplete_details': {'reason': 'max_output_tokens'}}
        response = SimpleNamespace(status='incomplete', model=MODEL, output_text='partial',
                                   model_dump=lambda **kwargs: raw)
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.pdf'
            source.write_bytes(b'synthetic source')
            args = SimpleNamespace(output=Path(tmp) / 'out', dpi=144, detail='high', overwrite=False)
            with patch('run_astra_ocr.rendered_pages', return_value=iter([Image.new('RGB', (20, 20))])):
                result = process_document(source, args, client, {'model': MODEL})
            self.assertEqual(result['error_type'], 'RuntimeError')
            self.assertEqual(len(list(args.output.rglob('incomplete-0001.json'))), 1)
            self.assertEqual(len(list(args.output.rglob('response-*.json'))), 0)
            self.assertEqual(len(list(args.output.rglob('source.json'))), 0)


if __name__ == '__main__':
    unittest.main()
