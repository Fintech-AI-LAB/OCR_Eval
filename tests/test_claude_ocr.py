"""Offline checks for gateway provenance and response failure handling."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_claude_ocr import MODEL, parse_response, process_document, safe_error_message, validate_checkpoint


class ClaudeTests(unittest.TestCase):
    def checkpoint(self):
        parsed = {'blocks': [{'type': 'paragraph', 'markdown': 'Account 0012X'}], 'uncertainties': []}
        return {'page_number': 1, **parsed, 'markdown': 'Account 0012X',
                'raw_response': {'model': MODEL, 'choices': [{'finish_reason': 'stop',
                    'message': {'content': json.dumps(parsed), 'refusal': None}}]}}

    def test_raw_text_is_preserved(self):
        record = self.checkpoint()
        validate_checkpoint(record, 1, MODEL)
        record['blocks'][0]['markdown'] = record['markdown'] = 'Account 00123'
        with self.assertRaisesRegex(ValueError, 'raw provider response'):
            validate_checkpoint(record, 1, MODEL)

    def test_wrong_identity_page_or_finish_reason(self):
        record = self.checkpoint()
        with self.assertRaises(ValueError):
            validate_checkpoint(record, 2, MODEL)
        with self.assertRaises(ValueError):
            validate_checkpoint(record, 1, 'other-model')
        for reason in ('length', 'content_filter', 'tool_calls'):
            record['raw_response']['choices'][0]['finish_reason'] = reason
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                validate_checkpoint(record, 1, MODEL)

    def test_ignored_schema_is_rejected(self):
        raw = self.checkpoint()['raw_response']
        raw['choices'][0]['message']['content'] = json.dumps({'page': 1, 'blocks': [], 'uncertainties': []})
        with self.assertRaisesRegex(ValueError, 'keys'):
            parse_response(raw, MODEL)

    def test_truncation_retains_raw_only(self):
        from PIL import Image
        record = self.checkpoint()
        record['raw_response']['choices'][0]['finish_reason'] = 'length'
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.pdf'; source.write_bytes(b'synthetic source')
            args = SimpleNamespace(output=Path(tmp) / 'out', dpi=144, overwrite=False)
            with patch('run_claude_ocr.rendered_pages', return_value=iter([Image.new('RGB', (20, 20))])), \
                 patch('run_claude_ocr.infer', return_value=record):
                result = process_document(source, args, None, {'model': MODEL})
            self.assertEqual(result['error_type'], 'ValueError')
            self.assertEqual(len(list(args.output.rglob('incomplete-0001.json'))), 1)
            self.assertFalse(list(args.output.rglob('response-*.json')))
            self.assertFalse(list(args.output.rglob('source.json')))

    def test_error_message_redacts_credentials(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'synthetic-secret'}):
            error = SimpleNamespace(body={'error': {'message': 'bad synthetic-secret sk-test***secret'}})
            self.assertEqual(safe_error_message(error), 'bad [redacted] [redacted]')

    def test_filtered_page_is_recorded_and_other_pages_continue(self):
        from PIL import Image
        class FilteredError(Exception):
            status_code = 400
            body = {'message': 'Output blocked by content filtering policy'}
        record = self.checkpoint()
        record.update(page_number=2, seconds=1)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.pdf'; source.write_bytes(b'synthetic source')
            args = SimpleNamespace(output=Path(tmp) / 'out', dpi=144, overwrite=False)
            with patch('run_claude_ocr.rendered_pages', return_value=iter([Image.new('RGB', (20, 20))] * 2)), \
                 patch('run_claude_ocr.infer', side_effect=[FilteredError(), record]):
                result = process_document(source, args, None, {'model': MODEL})
            self.assertEqual(result['error_type'], 'IncompleteDocument')
            self.assertEqual(result['completed_pages'], 1)
            self.assertEqual(result['failed_pages'][0]['page_number'], 1)
            self.assertEqual(len(list(args.output.rglob('response-0002.json'))), 1)
            self.assertFalse(list(args.output.rglob('source.json')))


if __name__ == '__main__':
    unittest.main()
