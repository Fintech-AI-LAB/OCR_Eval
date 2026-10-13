from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'standalone_ocr_eval')]
from ocr_eval.unified import agreement, signature, compare_saved, fixed_panel_scores
from evaluate_six_critical_fields import extract_pages


class UnifiedEvaluationTests(unittest.TestCase):
    def test_value_swap_reduces_order_sensitive_score(self):
        a = signature('Account 123 Amount 900')
        b = signature('Account 900 Amount 123')
        self.assertEqual(agreement(a, b)['token_dice'], 1)
        self.assertLess(agreement(a, b)['similarity'], 1)
        self.assertEqual(agreement(a, a)['similarity'], 1)

    def test_empty_failed_pages_never_earn_agreement(self):
        self.assertEqual(agreement(signature(''), signature(''))['similarity'], 0)
        self.assertEqual(agreement(signature(''), signature('words'))['similarity'], 0)

    def test_new_candidate_does_not_change_existing_scores(self):
        families = {'a':['A'], 'b':['B'], 'c':['C']}
        data = {'doc':{m:{'pages':[t], 'input_format':'plain'} for m,t in
                       [('a','Account 001 Amount 900'), ('b','Account 001 Amount 901'), ('c','Account 002 Amount 900')]}}
        before = compare_saved(data, families, ['a','b','c'])
        data['doc']['d'] = deepcopy(data['doc']['a'])
        families['d'] = ['A']
        after = compare_saved(data, families, ['a','b','c'])
        for method, score in before['scores'].items():
            self.assertEqual(score, after['scores'][method])

    def test_related_peers_have_one_family_vote_and_hybrid_excludes_both(self):
        families = {'candidate':['A','B'], 'a':['A'], 'b':['B'], 'c':['C'], 'd':['C'], 'e':['E']}
        pairs = {(a,b):0.5 for a in sorted(families) for b in sorted(families) if a<b}
        for peer, value in [('a',1),('b',1),('c',0.2),('d',0.4),('e',0.9)]:
            pairs[tuple(sorted(('candidate',peer)))] = value
        scores = fixed_panel_scores(pairs, families, ['a','b','c','d','e'])
        self.assertAlmostEqual(scores['candidate'], 60)

    def test_physical_page_swap_is_not_hidden_by_document_overlap(self):
        families = {m:[m] for m in 'abc'}
        data = {'doc':{m:{'pages':['Account 001','Amount 900'], 'input_format':'plain'} for m in families}}
        data['doc']['c']['pages'].reverse()
        page = compare_saved(data, families, list(families))
        document = compare_saved(data, families, list(families), whole_document=True)
        self.assertEqual(page['scores']['c'], 0)
        self.assertGreater(document['scores']['c'], page['scores']['c'])

    def test_empty_page_keeps_full_denominator_and_shared_mask_applies_to_all(self):
        families = {m:[m] for m in 'abc'}
        data = {'doc':{m:{'pages':['same',''], 'input_format':'plain'} for m in families}}
        full = compare_saved(data, families, list(families))
        subset = compare_saved(data, families, list(families), page_mask={'doc':[0]})
        self.assertEqual(full['scores']['a'], 50)
        self.assertEqual(subset['scores']['a'], 100)

    def test_unaligned_text_cannot_enter_page_track(self):
        families = {m:[m] for m in 'abc'}
        data = {'doc':{m:{'pages':None, 'document_text':'same', 'input_format':'plain'} for m in families}}
        with self.assertRaisesRegex(ValueError, 'alignment'):
            compare_saved(data, families, list(families))
        self.assertEqual(compare_saved(data, families, list(families), whole_document=True)['scores']['a'], 100)

    def test_no_sequence_alignment_is_computed(self):
        with patch('difflib.SequenceMatcher', side_effect=AssertionError('expensive alignment')):
            self.assertEqual(agreement(signature('one two'),signature('one two'))['similarity'], 1)

    def test_missing_first_invoice_does_not_use_later_page(self):
        invoice = 'DOC NO.: 0012\nDATE: 12.06.2025\nBill to address Customer\nYour customer account: 0099'
        pages = ['']*30
        pages[9] = invoice
        self.assertFalse(extract_pages('trade',pages,'plain'))
        pages[8] = invoice
        fields = extract_pages('trade',pages,'plain')
        self.assertEqual(len(fields),3)
        self.assertEqual({f['page'] for f in fields},{9})

    def test_adjacent_invoice_label_and_colon_value_keep_binding(self):
        pages = ['']*30
        pages[8] = 'DOC NO.\n: 0012\nCompany name\nDATE\n:12.06.2025\nBill to address Customer'
        extracted = extract_pages('trade',pages,'plain')
        self.assertEqual(len(extracted), 2)
        fields = {f['field']:f for f in extracted}
        self.assertEqual(fields['InvoiceNumber']['value'],'0012')
        self.assertEqual(fields['InvoiceDate']['value'],'12.06.2025')
        for f in fields.values():
            a,b = f['decoded_span']
            self.assertEqual(pages[8][a:b],f['value'])

    def test_duplicate_captures_remain_visible_to_scorer(self):
        pages = ['']*30
        pages[8] = 'DOC NO.: 0012\nDOC NO.: 9999\nDATE: 12.06.2025\nBill to address Customer'
        self.assertEqual([f['value'] for f in extract_pages('trade',pages,'plain') if f['field']=='InvoiceNumber'], ['0012','9999'])


if __name__ == '__main__':
    unittest.main()
