"""Corpus-run safeguards independent of the evaluated references and answers."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_six_critical_fields import extract


class PilotExtractionTests(unittest.TestCase):
    def test_distinct_sections_preserve_wrong_digits(self):
        text = ('With reference Current Account §0123, USD Domiciliary Account 456\n'
                'We attached the following\n'
                'That the company mandate Current Account 987, USD Domiciliary Account 654 '
                'be changed as follows')
        fields = {x['field']: x['value'] for x in extract('board', text)}
        self.assertEqual(fields['LetterCurrentAccount'], '§0123')
        self.assertEqual(fields['ResolutionCurrentAccount'], '987')

    def test_first_invoice_cannot_borrow_from_second(self):
        text = ('Bill to address : 100\nfirst invoice without header fields\n'
                'DOC NO. : 999\nDATE : 01.02.2030\nBill to address : 200\n'
                'Your customer account : 200')
        fields = {x['field']: x['value'] for x in extract('trade', text)}
        self.assertNotIn('InvoiceNumber', fields)
        self.assertNotIn('InvoiceDate', fields)
        self.assertNotIn('InvoiceCustomerAccount', fields)

    def test_maturity_binding_handles_wrap_but_rejects_wrong_upper_bound(self):
        text = 'More than 5 years but not more\t94%\nthan 10 years\nThresholds'
        fields = {x['field']: x['value'] for x in extract('csa', text)}
        self.assertEqual(fields['Valuation5To10Years'], '94%')
        changed = text.replace('10 years', '20 years')
        self.assertNotIn('Valuation5To10Years', {x['field'] for x in extract('csa', changed)})


if __name__ == '__main__':
    unittest.main()
