"""Regression cases for formatting tolerance, section binding and raw spans."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_six_critical_fields import extract
from ocr_eval.profiles import canonical_value


class SharedExtractorTests(unittest.TestCase):
    def fields(self, doc, text):
        fields = extract(doc, text)
        for item in fields:
            start, end = item['decoded_span']
            self.assertEqual(item['value'], text[start:end])
        self.assertEqual(len(fields), len({item['field'] for item in fields}))
        return {item['field']: item['value'] for item in fields}

    def test_signatories_allow_joined_honorific_and_bind_paragraphs(self):
        text = ('Kind Attn. Mr. Attention Name\nWith reference to the mandate\n'
                '1.Mr.Amit Agarwal\nCategory B\nPlease be informed of our addition\n'
                '1.Mr. Satya Narayan Patra\nCategory B\nWe attached the following')
        fields = self.fields('board', text)
        self.assertEqual(fields['LetterRemovedSignatory'], 'Amit Agarwal')
        self.assertEqual(fields['LetterAddedSignatory'], 'Satya Narayan Patra')

    def test_missing_removed_name_does_not_shift_added_name(self):
        text = ('With reference to the mandate\n[unreadable name]\n'
                'Please be informed of our addition\nMr. New Person\nCategory B\n'
                'We attached the following')
        fields = self.fields('board', text)
        self.assertNotIn('LetterRemovedSignatory', fields)
        self.assertEqual(fields['LetterAddedSignatory'], 'New Person')

    def test_signatory_paragraph_can_use_following_person_anchor(self):
        text = ('With reference to the mandate\nMr. Old Person\nCategory B\n'
                'Please he informed that the following person is authorized\n'
                'Mr.New Person\nCategory B\nWe attached the following')
        fields = self.fields('board', text)
        self.assertEqual(fields['LetterRemovedSignatory'], 'Old Person')
        self.assertEqual(fields['LetterAddedSignatory'], 'New Person')

    def test_letter_date_spacing_and_no_cross_section_fallback(self):
        text = 'September 13,2019\nWe attached the following\nOctober 14, 2020'
        self.assertEqual(self.fields('board', text)['LetterDate'], 'September 13,2019')
        self.assertNotIn('LetterDate', self.fields('board', text.split('\n', 1)[1]))

    def test_typed_month_date_accepts_comma_spacing_without_changing_literal_dates(self):
        for value in ['September 13,2019', 'Sep 13,2019', 'September 13, 2019']:
            with self.subTest(value=value):
                self.assertEqual(canonical_value(value, 'LetterDate', {'value_type': 'date'}),
                                 ('2019-09-13', None))
        self.assertEqual(canonical_value('February 30,2020', 'LetterDate', {'value_type': 'date'})[1],
                         'invalid_date')
        spec = {'value_type': 'date', 'comparison': 'literal'}
        self.assertEqual(canonical_value('September 13,2019', 'LetterDate', spec),
                         ('September 13,2019', None))

    def test_accounts_exclude_sentence_period_without_repairing_errors(self):
        text = ('Current Account 00123. USD Domiciliary Account §0045\n'
                'We attached the following\nThat the company mandate Current Account 00987. '
                'USD Domiciliary Account 8OO45 be changed as follows')
        fields = self.fields('board', text)
        self.assertEqual(fields['LetterCurrentAccount'], '00123')
        self.assertEqual(fields['LetterDomiciliaryAccount'], '§0045')
        self.assertEqual(fields['ResolutionCurrentAccount'], '00987')
        self.assertEqual(fields['ResolutionDomiciliaryAccount'], '8OO45')

    def test_schedule_heading_and_duplicate_empty_email_labels(self):
        text = ('Email: other@example.com\nGARBLED NOMINEES STAMP\nSCHEDULE\n'
                'Email:\nEmail finance@example.com\nEmail:\n')
        self.assertEqual(self.fields('board', text)['ScheduleEmail'], 'finance@example.com')
        self.assertNotIn('ScheduleEmail', self.fields('board', 'SCHEDULE\nEmail:\nName Someone'))

    def test_signing_rule_preserves_missing_parentheses(self):
        value = 'TWO SIGNATORIES ONE FROM EACH CATEGORY A &B MUST SIGN JOINTLY'
        self.assertEqual(self.fields('board', 'SCHEDULE\n' + value)['ScheduleSigningRule'], value)

    def test_csa_joined_date_label(self):
        self.assertEqual(self.fields('csa', 'dated as of7 March 2017')['AgreementDate'], '7 March 2017')

    def test_currency_and_amount_joined_or_separated(self):
        for value in ['USD38,973.63', 'USD 38,973.63', 'USD\n38,973.63']:
            with self.subTest(value=value):
                fields = self.fields('trade', 'Currency and Amount\n' + value + '\nRegistration Date and Time')
                self.assertEqual(fields['CoverCurrency'], 'USD')
                self.assertEqual(fields['CoverAmount'], '38,973.63')

    def test_missing_or_malformed_amount_does_not_capture_next_label_or_prefix(self):
        for value in ['USD\nRegistration Date and Time', 'USD38,973.63BAD']:
            with self.subTest(value=value):
                self.assertNotIn('CoverAmount', self.fields('trade', 'Currency and Amount\n' + value))
        self.assertNotIn('CoverCurrency', self.fields('trade', 'Currency and Amount USDX38.00'))

    def test_invoice_fields_do_not_fall_back_to_later_invoice(self):
        text = ('COMMERCIAL INVOICE\nBill to address first customer\n'
                'COMMERCIAL INVOICE\nDOC NO.: 99999\nDATE: 12.06.2025\n'
                'Bill to address second customer\nYour customer account: 77777')
        fields = self.fields('trade', text)
        self.assertNotIn('InvoiceNumber', fields)
        self.assertNotIn('InvoiceDate', fields)
        self.assertNotIn('InvoiceCustomerAccount', fields)

    def test_invoice_column_labels_bind_values_by_order_and_preserve_errors(self):
        text = ('Earlier text\n  DOC NO.\n\nDATE\nGST NO.\n\n : 0012O\n'
                ': 31.02.2041\n: G-009\nCOMMERCIAL INVOICE\nBill to address Someone')
        fields = self.fields('trade', text)
        self.assertEqual(fields['InvoiceNumber'], '0012O')
        self.assertEqual(fields['InvoiceDate'], '31.02.2041')
        # A changed value is captured unchanged, never selected from references.
        other = self.fields('trade', text.replace('0012O', '98765').replace('31.02.2041', '05.11.2039'))
        self.assertEqual(other['InvoiceNumber'], '98765')
        self.assertEqual(other['InvoiceDate'], '05.11.2039')

    def test_invoice_column_order_can_put_date_first(self):
        fields = self.fields('trade', 'DATE\nDOC NO.\n: 05.11.2039\n: 00098\nBill to address X')
        self.assertEqual(fields['InvoiceDate'], '05.11.2039')
        self.assertEqual(fields['InvoiceNumber'], '00098')

    def test_wrapped_individual_invoice_rows_still_bind_without_column_shift(self):
        fields = self.fields('trade', 'DOC NO.\n: 00098\nCompany Name\nDATE\n: 05.11.2039\nBill to address X')
        self.assertEqual(fields['InvoiceNumber'], '00098')
        self.assertEqual(fields['InvoiceDate'], '05.11.2039')

    def test_invoice_column_rejects_missing_extra_duplicate_or_unlabelled_rows(self):
        cases = ['DOC NO.\nDATE\nGST NO.\n: 123\n: 05.11.2039',
                 'DOC NO.\nDATE\n: 123\n: 05.11.2039\n: G-999',
                 'DOC NO.\nDATE\nDATE\n: 123\n: 05.11.2039\n: 06.11.2039',
                 'DOC NO.\nDATE\n123\n05.11.2039',
                 'DOC NO.\nDATE\nother text\n: 123\n: 05.11.2039']
        for header in cases:
            with self.subTest(header=header):
                fields = self.fields('trade', header + '\nBill to address X')
                self.assertNotIn('InvoiceNumber', fields)
                self.assertNotIn('InvoiceDate', fields)

    def test_invoice_column_does_not_recover_later_invoice_or_override_inline(self):
        later = 'DOC NO.\nDATE\n: 0098\n: 05.11.2039\nBill to address second customer'
        fields = self.fields('trade', 'COMMERCIAL INVOICE\nBill to address first customer\n' + later)
        self.assertNotIn('InvoiceNumber', fields)
        self.assertNotIn('InvoiceDate', fields)
        fields = self.fields('trade', 'DOC NO.: BAD\nDATE: BAD-DATE\n' + later)
        self.assertEqual(fields['InvoiceNumber'], 'BAD')
        self.assertEqual(fields['InvoiceDate'], 'BAD-DATE')


if __name__ == '__main__':
    unittest.main()
