"""Contract and source-grounding tests; no external API calls or credentials."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone_ocr_eval'))
from ocr_eval.llm_fields import ChatClient, compile_catalog, discover, text_blocks, read_json

CATALOG = {'schema_version': 1, 'schemas': [{'id': 'account', 'name': 'Account',
    'definition': 'A financial account', 'fields': [{'id': 'number', 'name': 'AccountNumber'}]}]}


class FakeClient:
    model = 'test-model'
    base_url = 'https://example.test/v1'
    def __init__(self, value='00123', evidence='Account: 00123', number=0, field='number'):
        self.calls = []
        self.value, self.evidence, self.number, self.field = value, evidence, number, field
    def ask(self, stage, payload):
        self.calls.append({'stage': stage, 'payload': payload})
        if stage == 'schema_selection':
            return {'selected_schema_ids': ['account'], 'reason': 'account context', 'uncovered_topics': []}
        return {'decisions': [{'field_id': self.field, 'critical': True,
            'reason': 'Determines account routing', 'status': 'found',
            'occurrences': [{'value': self.value, 'evidence': self.evidence, 'occurrence': self.number}]}]}


class LLMFieldsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
    def source(self, text, suffix='.md'):
        p = self.root / ('input' + suffix)
        p.write_bytes(text.encode())
        return p
    def test_markdown_grounding_crlf_unicode_and_leading_zeros(self):
        text = '€\r\nAccount: 00123'
        result = discover(self.source(text), CATALOG, FakeClient())
        f = result['fields'][0]
        self.assertEqual(text[f['location']['start']:f['location']['end']], '00123')
        self.assertEqual(len(result['api']['calls']), 2)
    def test_repaired_value_rejected(self):
        result = discover(self.source('Account: 00I23'), CATALOG, FakeClient())
        self.assertEqual(result['fields'], [])
        self.assertEqual(len(result['issues']), 1)
    def test_unknown_field_rejected(self):
        with self.assertRaisesRegex(ValueError, 'field ID'):
            discover(self.source('Account: 00123'), CATALOG, FakeClient(field='invented'))
    def test_unknown_schema_rejected(self):
        client = FakeClient()
        client.ask = lambda *a: {'selected_schema_ids': ['invented']}
        with self.assertRaisesRegex(ValueError, 'schema ID'):
            discover(self.source('Account: 00123'), CATALOG, client)
    def test_repeated_evidence_disambiguated(self):
        text = 'Account: 00123\nAccount: 00123'
        r = discover(self.source(text), CATALOG, FakeClient(number=1))
        self.assertEqual(r['fields'][0]['location']['start'], text.rindex('00123'))
    def test_json_pointer_escaping_and_decoded_offsets(self):
        p = self.source(json.dumps({'a/b~c': 'Account: 00123'}), '.json')
        r = discover(p, CATALOG, FakeClient())
        self.assertEqual(r['fields'][0]['location']['json_pointer'], '/a~1b~0c')
        self.assertEqual(r['fields'][0]['location']['start'], 9)
    def test_chunk_coverage(self):
        text = ''.join(str(i % 10) for i in range(355))
        blocks = text_blocks(self.source(text), 100, 20)
        positions = set()
        for b in blocks:
            self.assertEqual(b['text'], text[b['start']:b['start']+len(b['text'])])
            positions.update(range(b['start'], b['start']+len(b['text'])))
        self.assertEqual(len(positions), len(text))
    def test_unmapped_graph_fields_retained(self):
        graph = {'metadata': {}, 'nodes': [
            {'id': 'owner', 'type': 'BusinessComponent', 'name': 'Custom'},
            {'id': 'f', 'type': 'BusinessElement', 'name': 'CustomField',
             'parent_id': 'owner', 'mapping_status': 'unmapped', 'definition': 'A field'}], 'edges': []}
        p = self.source(json.dumps(graph), '.json')
        result = compile_catalog(p)
        self.assertEqual(result['schemas'][0]['fields'][0]['id'], 'f')
        self.assertEqual(len(result['ontology_sha256']), 64)
    def test_streamed_graph(self):
        p = self.source('{"metadata":{},"nodes":[\n' +
            json.dumps({'id':'s','type':'BusinessComponent','name':'S'}) + ',\n' +
            json.dumps({'id':'f','type':'BusinessElement','name':'F','parent_id':'s'}) +
            '\n],"edges":[\n]}', '.json')
        self.assertEqual(compile_catalog(p)['schemas'][0]['fields'][0]['id'], 'f')
    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_json(self.source('{"a":1,"a":2}', '.json'))
    def test_chat_request_and_trace_do_not_include_key(self):
        client = ChatClient('example-model', api_key='secret')
        response = {'id': 'response', 'model': 'example-model', 'usage': {'total_tokens': 1},
                    'choices': [{'finish_reason': 'stop', 'message': {'content': '{"ok":true}'}}]}
        with patch.object(client.opener, 'open', return_value=io.BytesIO(json.dumps(response).encode())) as mock:
            self.assertEqual(client.ask('test', {'task': 'Return JSON'}), {'ok': True})
        request = mock.call_args.args[0]
        self.assertTrue(request.full_url.endswith('/v1/chat/completions'))
        self.assertEqual(json.loads(request.data)['response_format'], {'type': 'json_object'})
        self.assertNotIn('secret', json.dumps(client.calls))
    def test_truncated_response_fails_closed(self):
        client = ChatClient('test', api_key='secret')
        response = {'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}
        with patch.object(client.opener, 'open', return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                client.ask('test', {})
    def test_context_does_not_count_as_evidence(self):
        p = self.source(json.dumps({'label': 'Account: 00123', 'amount': '25'}), '.json')
        result = discover(p, CATALOG, FakeClient())
        self.assertEqual(len(result['fields']), 1)
        self.assertEqual(result['fields'][0]['location']['json_pointer'], '/label')
        self.assertEqual(len(result['issues']), 1)
    def test_unlocated_and_noncritical_preserved(self):
        client = FakeClient()
        original = client.ask
        def ask(stage, payload):
            if stage == 'schema_selection':
                return original(stage, payload)
            return {'decisions': [{'field_id': 'number', 'critical': False,
                'reason': 'Administrative example only', 'status': 'unlocated', 'occurrences': []}]}
        client.ask = ask
        result = discover(self.source('Account form'), CATALOG, client)
        self.assertFalse(result['decisions'][0]['critical'])
        self.assertEqual(result['fields'], [])
    def test_router_can_abstain(self):
        client = FakeClient()
        client.ask = lambda *a: {'selected_schema_ids': [], 'reason': 'No relevant schema', 'uncovered_topics': ['signatures']}
        result = discover(self.source('signature'), CATALOG, client)
        self.assertEqual(result['fields'], [])
        self.assertEqual(result['routes'][0]['uncovered_topics'], ['signatures'])
    def test_compact_catalog_keeps_source_refs(self):
        data = {'concepts': {'Account': {'id': 'concept:Account', 'fields': ['Number'],
            'field_evidence': [{'id': 'src:1', 'name': 'Number'}]}}}
        result = compile_catalog(self.source(json.dumps(data), '.json'))
        self.assertEqual(result['schemas'][0]['fields'][0]['source_refs'], ['src:1'])
    def test_cli_workflow(self):
        from ocr_eval.cli import main
        source = self.source('Account: 00123')
        catalog = self.root / 'catalog.json'
        catalog.write_text(json.dumps(CATALOG))
        output = self.root / 'result.json'
        with patch('ocr_eval.llm_fields.ChatClient', return_value=FakeClient()):
            main(['identify-critical-fields', str(source), '--schemas', str(catalog),
                  '--model', 'test', '--output', str(output)])
        self.assertEqual(read_json(output)['fields'][0]['raw_value'], '00123')

if __name__ == '__main__':
    unittest.main()
