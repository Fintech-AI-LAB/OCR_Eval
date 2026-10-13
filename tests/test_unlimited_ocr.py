"""Reject stale/incomplete OCR and check the upstream repetition guard."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_unlimited_ocr import clean_markdown, banned_tokens, validate_response, METHOD, MODEL, REVISION, sha, orient_page
from evaluate_unlimited_ocr import verify_unlimited
from ocr_common import destination_for, write_json


class UnlimitedTests(unittest.TestCase):
    def test_cleanup_retains_bad_digits_and_text_blocks(self):
        raw = ('<|det|>text [0, 1, 2, 3]<|/det|>Account 00129\nnext line\n'
               '<|det|>image [0, 0, 1, 1]<|/det|>image.png\n'
               '<|det|>table<|/det|><table><tr><td>9.01</td></tr></table>')
        self.assertEqual(clean_markdown(raw), 'Account 00129\nnext line\n\n<table><tr><td>9.01</td></tr></table>')

    def test_ngram_guard_and_sliding_window(self):
        self.assertEqual(banned_tokens([1, 2, 3, 1, 2], 3, 128), {3})
        self.assertEqual(banned_tokens([1, 2, 3, 9, 9, 1, 2], 3, 3), set())
        self.assertEqual(banned_tokens([1, 2], 3, 128), set())
        self.assertEqual(banned_tokens([1, 2, 3], 1, 2), {2, 3})

    def test_rotation_direction_preserves_pixels(self):
        from PIL import Image
        image = Image.new('RGB', (2, 3))
        image.putdata([(i, i, i) for i in range(6)])
        orientation = {'documents': {'sample': {'input_sizes': [[2, 3]], 'angles_ccw': [270]}}}
        rotated, angle = orient_page(image, 'sample', 1, orientation)
        self.assertEqual(angle, 270)
        self.assertEqual(rotated.size, (3, 2))
        self.assertEqual(list(rotated.tobytes()[::3]), [4, 2, 0, 5, 3, 1])
        self.assertEqual(rotated.transpose(Image.Transpose.ROTATE_90).tobytes(), image.tobytes())
        orientation['documents']['sample']['input_sizes'] = [[3, 2]]
        with self.assertRaisesRegex(ValueError, 'dimensions mismatch'):
            orient_page(image, 'sample', 1, orientation)

    def fixture(self, root, rotated=False):
        from PIL import Image
        source = root / 'synthetic.pdf'
        source.write_bytes(b'synthetic PDF bytes; renderer mocked')
        output = root / 'output'
        config = {'model': MODEL, 'revision': REVISION, 'backend': METHOD, 'dpi': 144}
        image = Image.new('RGB', (2, 3))
        image.putdata([(i, i, i) for i in range(6)])
        if rotated:
            config['orientation'] = {'manifest': str(root / 'manifest.json'), 'documents': {
                source.name: {'input_sizes': [[2, 3]], 'angles_ccw': [270]}}}
        transformed, angle = orient_page(image, source.name, 1, config.get('orientation'))
        folder = destination_for(source, root, output, config)
        response = {'page': 1, 'image_sha256': hashlib.sha256(transformed.tobytes()).hexdigest(),
            'text': 'Account 00129', 'markdown': 'Account 00129', 'finish_reason': 'stop', 'truncated': False}
        if rotated:
            response.update(original_image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                            rotation_ccw_degrees=angle, image_size=list(transformed.size))
        write_json(folder / 'response-0001.json', response)
        (folder / 'document.md').write_text(response['markdown'])
        write_json(folder / 'source.json', {'source': source.name, 'source_sha256': sha(source),
                                          'config': config, 'status': 'complete', 'responses': 1})
        record = {'source': source.name, 'source_sha256': sha(source), 'run': str(folder.relative_to(output)),
                  'pages': 1, 'artifact_sha256': {p.name: sha(p) for p in folder.iterdir()}}
        return source, output, folder, image, response, record

    def test_verifier_reconstructs_rotated_pixels_and_rejects_wrong_angle(self):
        with tempfile.TemporaryDirectory() as temp:
            source, root, folder, image, response, record = self.fixture(Path(temp), rotated=True)
            orientation = json.loads((folder / 'source.json').read_text())['config']['orientation']
            with patch('evaluate_unlimited_ocr.pages', return_value=[image]), \
                    patch('evaluate_unlimited_ocr.load_orientation', return_value=orientation):
                verify_unlimited(record, root, source, 1)
                response['rotation_ccw_degrees'] = 90
                write_json(folder / 'response-0001.json', response)
                record['artifact_sha256']['response-0001.json'] = sha(folder / 'response-0001.json')
                with self.assertRaisesRegex(ValueError, 'Rotation metadata mismatch'):
                    verify_unlimited(record, root, source, 1)

    def test_hashes_pixels_and_page_completeness(self):
        with tempfile.TemporaryDirectory() as temp:
            source, root, folder, image, response, record = self.fixture(Path(temp))
            with patch('evaluate_unlimited_ocr.pages', return_value=[image]):
                verify_unlimited(record, root, source, 1)
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    verify_unlimited(record, root, source, 2)
                response['image_sha256'] = 'changed image'
                write_json(folder / 'response-0001.json', response)
                record['artifact_sha256']['response-0001.json'] = sha(folder / 'response-0001.json')
                with self.assertRaisesRegex(ValueError, 'Rendered image'):
                    verify_unlimited(record, root, source, 1)

    def test_reject_tampered_text_and_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            source, root, folder, image, response, record = self.fixture(Path(temp))
            response['markdown'] = 'manually fixed 00123'
            with self.assertRaisesRegex(ValueError, 'Cleaned text'):
                validate_response(response, 1)
            response['markdown'] = response['text']
            response['finish_reason'] = 'length'
            response['truncated'] = True
            with self.assertRaisesRegex(ValueError, 'Incomplete generation'):
                validate_response(response, 1)
            source.write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'Source hash'):
                verify_unlimited(record, root, source, 1)


if __name__ == '__main__':
    unittest.main()
