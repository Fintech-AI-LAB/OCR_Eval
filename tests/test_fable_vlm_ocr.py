"""Offline tests for the claude-fable-5-1 VLM runner and evaluator (no API calls)."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_fable_vlm_ocr as runner
from ocr_common import destination_for

ACCEPTED = [runner.MODEL]


def final_message(n, text=None, stop='end_turn', model=runner.MODEL, content=None):
    if text is None:
        text = json.dumps({'transcription': f'<table><tr><td>00123</td><td>-5.00</td></tr></table>\nPage {n}',
                           'uncertainties': [] if n % 2 else ['digit after 00 faint']})
    return {'id': f'msg_{n}', 'type': 'message', 'role': 'assistant', 'model': model,
            'content': content if content is not None else
            [{'type': 'thinking', 'thinking': '', 'signature': 'sig'}, {'type': 'text', 'text': text}],
            'stop_reason': stop, 'stop_sequence': None, 'usage': {'input_tokens': 2600, 'output_tokens': 900}}


def events_for(final, with_stop=True):
    events = [{'type': 'message_start', 'message': {'id': final['id'], 'model': final['model']}},
              {'type': 'message_delta', 'delta': {'stop_reason': final['stop_reason']}, 'usage': {'output_tokens': 900}}]
    return events + ([{'type': 'message_stop'}] if with_stop else [])


class FakeModel:
    """Stands in for call_model; `script` maps call number -> kind."""

    serial = 0  # unique response ids across fakes, like the real API

    def __init__(self, script=None):
        self.script, self.calls = script or {}, 0

    def __call__(self, client, request):
        self.calls += 1
        FakeModel.serial += 1
        assert request['model'] == runner.MODEL and request['messages'][0]['content'][1]['text'] == runner.PROMPT
        kind = self.script.get(self.calls, 'ok')
        if kind == 'network':
            raise ConnectionError('simulated network failure')
        n = FakeModel.serial
        if kind == 'refusal':
            final = final_message(n, stop='refusal', content=[])
        elif kind == 'truncated':
            final = final_message(n, stop='max_tokens', content=[{'type': 'text', 'text': '{"transcr'}])
        else:
            final = final_message(n)
        return final, events_for(final), {'request_id': f'req_{self.calls}', 'http_status': 200}, \
            {'started_at': 't0', 'finished_at': 't1', 'seconds': 1.25, 'first_event_seconds': 0.2}


def make_pdf(path, count=2):
    from PIL import Image
    images = [Image.new('RGB', (200, 260), 'white') for _ in range(count)]
    images[0].save(path, 'PDF', resolution=72.0, save_all=True, append_images=images[1:])


class ClassificationTests(unittest.TestCase):
    def test_completed(self):
        final = final_message(1)
        outcome, flags, payload = runner.classify(final, events_for(final), ACCEPTED)
        self.assertEqual((outcome, flags), ('completed', []))
        assert payload is not None
        self.assertIn('00123', payload['transcription'])  # leading zeros kept verbatim

    def test_flagged_responses_never_complete(self):
        cases = {
            'refusal': final_message(1, stop='refusal', content=[]),
            'truncated': final_message(1, stop='max_tokens'),
            'model_mismatch': final_message(1, model='claude-opus-5-5'),
            'invalid_output': final_message(1, text='not json'),
        }
        for expected, final in cases.items():
            outcome, flags, payload = runner.classify(final, events_for(final), ACCEPTED)
            self.assertEqual(outcome, expected, flags)
            self.assertIsNone(payload)

    def test_schema_and_stream_integrity(self):
        extra = final_message(1, text=json.dumps({'transcription': 'x', 'uncertainties': [], 'note': 'y'}))
        self.assertEqual(runner.classify(extra, events_for(extra), ACCEPTED)[0], 'invalid_output')
        final = final_message(1)
        self.assertEqual(runner.classify(final, events_for(final, with_stop=False), ACCEPTED)[0], 'incomplete')


class RequestTests(unittest.TestCase):
    def test_request_and_redaction(self):
        request = runner.build_request(b'\x89PNG fake')
        image = request['messages'][0]['content'][0]
        self.assertEqual(image['transformations'], {'oversized_image': 'error'})
        self.assertEqual(request['extra_body']['thinking'], {'type': 'adaptive'})
        self.assertEqual(request['extra_body']['output_config']['effort'], 'high')
        for key in ('temperature', 'top_p', 'top_k', 'system'):
            self.assertNotIn(key, request)
        body = runner.redacted_request(request, b'\x89PNG fake')
        self.assertNotIn('data', body['messages'][0]['content'][0]['source'])
        self.assertTrue(body['stream'])
        self.assertIn('output_config', body)

    def test_prompt_is_the_specified_instruction(self):
        self.assertTrue(runner.PROMPT.startswith('Transcribe every visible text region'))
        self.assertTrue(runner.PROMPT.endswith('Do not include explanatory commentary in the transcription.'))
        self.assertIn('Preserve table rows and columns using HTML tables.', runner.PROMPT)

    def test_cache_identity_covers_source_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'a.pdf'
            source.write_bytes(b'one')
            config = runner.inference_config()
            key = runner.cache_key(source, config)
            self.assertEqual(destination_for(source, root, root / 'out', config).name, key)
            for change in ({'model': 'other'}, {'prompt': runner.PROMPT + ' '}, {'dpi': 150}, {'max_tokens': 1}):
                self.assertNotEqual(runner.cache_key(source, {**config, **change}), key)
            source.write_bytes(b'two')
            self.assertNotEqual(runner.cache_key(source, config), key)

    def test_call_model_keeps_raw_events_only(self):
        class Event(SimpleNamespace):
            def model_dump(self, **_):
                return dict(vars(self))

        final = final_message(1)

        class Stream:
            response = SimpleNamespace(headers={'request-id': 'req_x'}, status_code=200,
                                       request=SimpleNamespace(headers={'anthropic-version': '2023-06-01',
                                                                        'x-api-key': 'sk-ant-secret'}))

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                yield Event(type='message_start', message={'id': 'msg_1'})
                yield Event(type='text', text='duplicate')
                yield Event(type='content_block_stop', index=0, content_block={'snapshot': True})
                yield Event(type='message_stop', message={'snapshot': True})

            def get_final_message(self):
                return SimpleNamespace(model_dump=lambda **_: final)

        client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **_: Stream()))
        got, events, http, timing = runner.call_model(client, runner.build_request(b'png'))
        self.assertEqual(got, final)
        self.assertEqual([e['type'] for e in events], ['message_start', 'content_block_stop', 'message_stop'])
        self.assertEqual(events[1], {'type': 'content_block_stop', 'index': 0})
        self.assertEqual(http['request_id'], 'req_x')
        self.assertNotIn('sk-ant', json.dumps(http))
        self.assertIn('seconds', timing)


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'doc.pdf'
        make_pdf(self.source)
        self.args = SimpleNamespace(output=self.root / 'out', retry_incomplete=False, save_images=True)
        self.context = {'config': runner.inference_config(), 'accepted_models': ACCEPTED,
                        'sdk': {'package': 'anthropic', 'version': 'test'},
                        'preflight': {'path': 'preflight/p.json', 'sha256': 'x'}, 'workers': 1}
        self.original = runner.call_model

    def tearDown(self):
        runner.call_model = self.original
        self.tmp.cleanup()

    def run_doc(self, fake, **args):
        runner.call_model = fake
        for key, value in args.items():
            setattr(self.args, key, value)
        return runner.process_document(self.source, 2, self.args, None, self.context, threading.Event())

    def test_complete_run_resumes_without_new_requests(self):
        fake = FakeModel()
        result = self.run_doc(fake)
        self.assertTrue(result['complete'])
        run = self.args.output.resolve() / result['run']
        record = json.loads((run / 'pages/page-0001.json').read_text())
        self.assertEqual(record['render']['width'], 400)  # 200 pt at 144 DPI
        self.assertEqual((run / 'pages/page-0001.md').read_text(), record['transcription'])
        self.assertEqual(record['response']['returned_model'], runner.MODEL)
        self.assertIn('<!-- page: 2 -->', (run / 'document.md').read_text())
        self.assertNotIn('digit after 00 faint', (run / 'document.md').read_text())
        self.assertIn('digit after 00 faint', (run / 'uncertainties.md').read_text())
        self.assertTrue((run / record['raw_events_path']).exists())
        gzip.decompress((run / record['raw_events_path']).read_bytes())
        again = self.run_doc(fake)
        self.assertTrue(again['complete'])
        self.assertEqual(fake.calls, 2)
        self.assertEqual(again['manifest_record'], result['manifest_record'])

    def test_refusal_is_kept_and_not_silently_retried(self):
        result = self.run_doc(FakeModel({2: 'refusal'}))
        self.assertFalse(result['complete'])
        self.assertEqual(result['pages'], {'1': 'completed', '2': 'refusal'})
        run = self.args.output.resolve() / result['run']
        self.assertFalse((run / 'pages/page-0002.json').exists())
        self.assertTrue((run / 'attempts/page-0002-attempt-01.json').exists())
        fake = FakeModel()
        self.assertFalse(self.run_doc(fake)['complete'])
        self.assertEqual(fake.calls, 0)
        retried = self.run_doc(fake, retry_incomplete=True)
        self.assertTrue(retried['complete'])
        self.assertTrue((run / 'attempts/page-0002-attempt-02.json').exists())
        self.assertTrue((run / 'raw/page-0002-attempt-01.response.json').exists())

    def test_truncation_and_transport_errors_are_not_completed(self):
        result = self.run_doc(FakeModel({1: 'truncated'}))
        self.assertEqual(result['pages']['1'], 'truncated')
        self.assertFalse(result['complete'])
        self.args.retry_incomplete = True
        stopped = self.run_doc(FakeModel({1: 'network'}))
        self.assertEqual(stopped['pages']['1'], 'transport_error')
        self.assertFalse(stopped['complete'])

    def test_tampering_is_detected(self):
        result = self.run_doc(FakeModel())
        run = self.args.output.resolve() / result['run']
        (run / 'pages/page-0001.md').write_text('edited')
        with self.assertRaisesRegex(ValueError, 'Markdown'):
            self.run_doc(FakeModel())

    def test_evaluator_verifies_selected_run(self):
        import evaluate_fable_vlm as evaluator
        result = self.run_doc(FakeModel())
        record = result['manifest_record']
        root = self.args.output.resolve()
        records, _ = evaluator.verify_run(record, self.source, root, 2, ACCEPTED)
        self.assertEqual([r['page_number'] for r in records], [1, 2])
        raw = root / record['run'] / records[0]['raw_response_path']
        original = raw.read_text()
        tampered = original.replace('Page ', 'Tampered page ')
        self.assertNotEqual(original, tampered)
        raw.write_text(tampered)
        with self.assertRaises(ValueError):
            evaluator.verify_run(record, self.source, root, 2, ACCEPTED)


class DiagnosisTests(unittest.TestCase):
    def test_diagnosis_categories(self):
        import evaluate_fable_vlm as evaluator
        d = evaluator.diagnose
        self.assertEqual(d('ACC 00123', None, 'x ACC  00123 y', '')[0], 'parser_or_field_binding_failure')
        self.assertEqual(d('1,000.00', None, 'amount 1000.00', '')[0], 'format_or_punctuation_difference')
        self.assertEqual(d('00123456', '00128456', 'nothing', '')[0], 'ocr_transcription_error_near_miss')
        self.assertEqual(d('00123456', None, 'nothing', 'page 3 has 00123456')[0], 'value_only_on_another_page')
        self.assertEqual(d('00123456', None, 'nothing', 'nothing')[0], 'ocr_omission_or_misread')
        self.assertEqual(d('-5.00', None, 'amount 5.00', '')[0], 'ocr_omission_or_misread')  # lost sign is not formatting


if __name__ == '__main__':
    unittest.main()
