import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'standalone_ocr_eval'))
from field_values import extract_field_values, field_value_metrics
from cross_model_eval import evaluate
from score_page import score_page, score_page_details, score_models
from evaluate_outputs import compare_page
import ocr_eval
from ocr_eval.cli import main
from rank_ocr import calculate, ENGINES


class FieldValueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ontology = self.root / 'ontology.json'
        self.write_ontology({'Amount': {}, 'Currency': {}, 'Account': {},
                             'SettlementDate': {}, 'Rate': {}, 'IBAN': {},
                             'Company': {'aliases': ['Business Name'], 'value_type': 'text'},
                             'Invoice': {'fields': ['Reference', 'Amount'],
                                         'field_types': {'Reference': 'identifier'}},
                             'Payment': {'fields': ['Reference']}})

    def write_ontology(self, concepts, **settings):
        self.ontology.write_text(json.dumps({'concepts': concepts, **settings}))

    def extract(self, text, fmt='plain'):
        return extract_field_values(text, self.ontology, fmt)

    def compare(self, a, b, fmt='plain'):
        return field_value_metrics(self.extract(a, fmt), self.extract(b, fmt))

    def files(self, *texts, suffix='.txt'):
        paths = [self.root / f'{i}{suffix}' for i in range(len(texts))]
        for path, text in zip(paths, texts):
            path.write_text(text)
        return paths

    def data(self, *texts, fmt='plain'):
        return {'doc': {str(i): {'pages': [t], 'input_format': fmt} for i, t in enumerate(texts)}}

    def test_swapping_field_values_is_not_full_agreement(self):
        a, b = 'Amount 100 Currency USD Account 200', 'Amount 200 Currency USD Account 100'
        row = compare_page('demo', 1, 'a', 'b', a, b, self.ontology, ontology_mode='values')
        self.assertEqual(row['ontology_weighted_overlap'], 1)
        self.assertAlmostEqual(row['ontology_field_value_agreement'], 1 / 3)
        self.assertEqual(row['per_field_agreement']['Amount']['agreement'], 0)
        self.assertAlmostEqual(score_page(self.files(a, b), ontology_path=self.ontology, ontology_mode='values'), 100 / 3)

    def test_wrong_value_does_not_receive_label_credit(self):
        self.assertEqual(self.compare('Amount 100', 'Amount 900')['ontology_field_value_agreement'], 0)
        self.assertEqual(score_page(self.files('Amount 100', 'Amount 900'), ontology_path=self.ontology, ontology_mode='values'), 0)

    def test_canonical_names_and_explicit_aliases(self):
        for a, b in [('SettlementDate: 2026-10-01', 'Settlement Date: 1 October 2026'),
                     ('Company: Acme Ltd', 'Business Name: ACME   LTD')]:
            self.assertEqual(self.compare(a, b)['ontology_field_value_agreement'], 1)

    def test_identifier_zeroes_and_amount_sign_currency_and_units(self):
        for a, b in [('Account: 00123', 'Account: 123'), ('Amount: -100', 'Amount: 100'),
                     ('Amount: 100 USD', 'Amount: 100 EUR'), ('Rate: 5%', 'Rate: 5'),
                     ('Amount: 1.234', 'Amount: 1,234')]:
            self.assertEqual(self.compare(a, b)['ontology_field_value_agreement'], 0)
        for a, b in [('Amount: 1,000.00', 'Amount: 1000'), ('Amount: (100.00)', 'Amount: -100'),
                     ('IBAN: GB82 WEST 1234', 'IBAN: GB82WEST1234')]:
            self.assertEqual(self.compare(a, b)['ontology_field_value_agreement'], 1)
        digits = '123456789012345678901234567890123456789'
        self.assertEqual(self.compare('Amount: (' + digits + ')', 'Amount: -' + digits)['ontology_field_value_agreement'], 1)

    def test_separate_currency_is_associated_with_amount(self):
        row = self.compare('Amount: 100\nCurrency: USD', 'Amount: 100\nCurrency: EUR')
        self.assertEqual(row['per_field_agreement']['Amount']['agreement'], 0)
        self.assertEqual(row['ontology_field_value_agreement'], 0)

    def test_ambiguous_and_invalid_dates_are_not_guessed_or_rewarded(self):
        row = self.compare('Settlement Date: 01/02/2026', 'Settlement Date: 01/02/2026')
        self.assertEqual(row['ontology_field_value_agreement'], 0)
        self.assertEqual(row['left_ontology_fields'][0]['error'], 'ambiguous_date')
        self.assertEqual(self.compare('SettlementDate: 2026-02-30', 'SettlementDate: 2026-02-30')['ontology_field_value_agreement'], 0)
        self.write_ontology({'SettlementDate': {}}, date_order='dmy')
        self.assertEqual(self.compare('SettlementDate: 01/02/2026', 'Settlement Date: 2026-02-01')['ontology_field_value_agreement'], 1)

    def test_missing_repeated_and_invalid_values(self):
        row = self.compare('Amount: 100\nAccount: 001', 'Amount: 100')
        self.assertAlmostEqual(row['ontology_field_value_agreement'], 2 / 3)
        self.assertEqual(row['unmatched_left_ontology_fields'][0]['field'], 'Account')
        self.assertEqual(self.compare('Amount: 100\nAmount: 200', 'Amount: 200\nAmount: 100')['ontology_field_value_agreement'], 0)
        self.assertAlmostEqual(self.compare('Amount: 100\nAmount: 100', 'Amount: 100')['ontology_field_value_agreement'], 2 / 3)
        for value in ['', 'N/A', 'xyz', '<REDACTED>']:
            self.assertEqual(self.compare('Amount: ' + value, 'Amount: ' + value)['ontology_field_value_agreement'], 0)

    def test_ambiguous_owners_require_context(self):
        extraction = self.extract('Reference: 001')
        self.assertFalse(extraction['fields'])
        self.assertEqual(extraction['issues'][0]['reason'], 'ambiguous_field')
        self.assertEqual(self.compare('Invoice\nReference: 001', 'Payment\nReference: 001')['ontology_field_value_agreement'], 0)
        self.assertEqual(self.extract('Invoice\nReference: 001')['fields'][0]['field'], 'Invoice.Reference')

    def test_unlabelled_values_and_prose_are_not_extracted(self):
        self.assertFalse(self.extract('100 USD 00123 2026-10-01')['fields'])
        self.assertFalse(self.extract('The amount of work is substantial')['fields'])
        self.assertFalse(self.extract('Amount due to us is documented')['fields'])
        self.assertFalse(self.extract('Account numbers for 2 customers')['fields'])
        self.assertIsNone(self.compare('Amount', 'Amount')['ontology_field_value_agreement'])

    def test_labels_on_preceding_line_and_vocabulary_inside_text_value(self):
        self.assertEqual(self.compare('Amount\n100', 'Amount: 100')['ontology_field_value_agreement'], 1)
        self.assertEqual(self.compare('Account:\n00123', 'Account: 00123')['ontology_field_value_agreement'], 1)
        self.assertEqual(self.extract('Company: Acme Account Ltd')['fields'][0]['raw_value'], 'Acme Account Ltd')

    def test_markdown_key_value_table_matches_plain_fields(self):
        md = '| Field | Value |\n|---|---|\n| Amount | 100.00 |\n| Account | 00123 |'
        paths = self.files(md, 'Amount: 100\nAccount: 00123', suffix='.md')
        self.assertEqual(score_page(paths, ontology_path=self.ontology, ontology_mode='values'), 100)

    def test_table_rows_preserve_field_and_record_associations(self):
        a = '| Item | Amount | Account |\n|---|---|---|\n| X | 100 | 001 |\n| Y | 200 | 002 |'
        b = '| Item | Amount | Account |\n|---|---|---|\n| X | 200 | 001 |\n| Y | 100 | 002 |'
        row = self.compare(a, b, 'markdown')
        self.assertEqual(row['ontology_field_value_agreement'], 0.5)
        reordered = '| Item | Amount | Account |\n|---|---|---|\n| Y | 200 | 002 |\n| X | 100 | 001 |'
        self.assertEqual(self.compare(a, reordered, 'markdown')['ontology_field_value_agreement'], 1)

    def test_unknown_table_columns_and_empty_values(self):
        table = '| Item | Description | Amount |\n|---|---|---|\n| A | Service | 100 |\n| B | Goods | |'
        fields = self.extract(table, 'markdown')['fields']
        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0]['record'], 'a')
        self.assertEqual(fields[1]['error'], 'missing_value')

    def test_unicode_alias_offsets(self):
        self.write_ontology({'Street': {'aliases': ['Straße']}})
        self.assertEqual(self.extract('Straße: Main Road')['fields'][0]['raw_value'], 'Main Road')

    def test_html_tables_and_plain_literal_markup(self):
        html = '<table><tr><th>Amount</th><th>Account</th></tr><tr><td>100</td><td>001</td></tr></table>'
        md = '| Amount | Account |\n|---|---|\n| 100 | 001 |'
        a, b = self.files(html, md, suffix='.md')
        self.assertEqual(score_page([a, b], ontology_path=self.ontology, ontology_mode='values'), 100)
        self.assertEqual(self.extract('Company: <p> literal')['fields'][0]['raw_value'], '<p> literal')

    def test_batch_page_and_standalone_scores_match(self):
        texts = ['Amount: 100\nAccount: 001', 'Account: 001\nAmount: 900', 'Amount: 100\nAccount: 001']
        paths = self.files(*texts)
        result, pairs, volumes, normalized = evaluate(self.data(*texts), ontology_path=self.ontology, ontology_mode='values')
        details = score_page_details(paths, ontology_path=self.ontology, ontology_mode='values')
        scores = {r['model']: r['consensus_score'] for r in result['ranking']}
        self.assertEqual(details['model_scores'], [scores[str(i)] for i in range(3)])
        self.assertEqual(score_models(paths, ontology_path=self.ontology, ontology_mode='values'), details['model_scores'])
        self.assertEqual(ocr_eval.score_page_details(paths, ontology_path=self.ontology, ontology_mode='values'), details)
        self.assertEqual(ocr_eval.evaluate(self.data(*texts), ontology_path=self.ontology, ontology_mode='values'), (result, pairs, volumes, normalized))
        self.assertEqual(result['settings']['primary_metric'], 'ontology_field_value_agreement')
        self.assertEqual(evaluate(normalized, ontology_path=self.ontology, ontology_mode='values')[0]['ranking'], result['ranking'])
        with patch('score_page.compare_page', side_effect=AssertionError('Unnecessary alignment')):
            self.assertEqual(score_models(paths, ontology_path=self.ontology, ontology_mode='values'), details['model_scores'])

    def test_normalized_table_replay_preserves_structure(self):
        md = '| Item | Amount |\n|---|---|\n| A | 100 |\n| B | 200 |'
        original = evaluate(self.data(md, md, md, fmt='markdown'), ontology_path=self.ontology, ontology_mode='values')
        replay = evaluate(json.loads(json.dumps(original[3])), ontology_path=self.ontology, ontology_mode='values')
        self.assertEqual(original, replay)

    def test_no_fields_excluded_and_empty_peers_score_zero(self):
        data = self.data('Amount: 100', '', '')
        for record in data['doc'].values():
            record['pages'].append('ordinary prose')
        result, pairs, _, _ = evaluate(data, ontology_path=self.ontology, ontology_mode='values')
        self.assertEqual(len(result['excluded_pages']), 1)
        self.assertTrue(all(p['primary_similarity'] == 0 for p in pairs))
        with self.assertRaisesRegex(ValueError, 'No scorable pages'):
            evaluate(self.data('ordinary prose', 'ordinary prose', 'ordinary prose'), ontology_path=self.ontology, ontology_mode='values')
        with self.assertRaisesRegex(ValueError, 'No ontology field values'):
            score_page(self.files('Amount', 'Amount'), ontology_path=self.ontology, ontology_mode='values')

    def test_saved_value_metrics_cannot_be_ranked_as_region_scores(self):
        texts = ['Amount: 100', 'Amount: 100', 'Amount: 900', 'Amount: 900']
        rows = [compare_page('doc', 1, a, b, texts[i], texts[j], self.ontology, ontology_mode='values')
                for i, a in enumerate(ENGINES) for j, b in enumerate(ENGINES) if i < j]
        with self.assertRaisesRegex(ValueError, 'current ontology region'):
            calculate(rows)

    def test_cli_and_legacy_mode(self):
        paths = self.files('Amount: 100', 'Amount: 900')
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(['page', *map(str, paths), '--ontology', str(self.ontology), '--ontology-mode', 'values'])
        result = json.loads(output.getvalue())
        self.assertEqual(result['score'], 0)
        self.assertEqual(score_page(paths, ontology_path=self.ontology, ontology_mode='mentions'), 75)
        with self.assertRaises(ValueError):
            score_page(paths, ontology_path=self.ontology, ontology_mode='bad')

    def test_invalid_schema_fails(self):
        for record in [{'value_type': 'bogus'}, {'aliases': 'Alias'}, {'field_types': {'X': 'number'}}, {'date_order': 'guess'}]:
            self.write_ontology({'Amount': record})
            with self.assertRaises(ValueError):
                self.extract('Amount: 100')


if __name__ == '__main__':
    unittest.main()
