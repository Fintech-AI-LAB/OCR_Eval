from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone_ocr_eval'))
from ocr_eval.corroboration import corroborate_saved, corroboration_policy, passage_csv_rows


def evaluate(texts, families=None, panel=None, statuses=None, **options):
    families = families or {m: [m] for m in texts}
    records = {m: {'pages': [text], 'input_format': 'plain',
                   'statuses': [(statuses or {}).get(m, 'complete')]} for m, text in texts.items()}
    return corroborate_saved({'doc': records}, families, panel or list(texts), **options)


def passages(result, method='a'):
    return [r for r in result['passages'] if r['method'] == method]


class CorroborationTests(unittest.TestCase):
    def test_identical_independent_families_support_each_passage(self):
        result = evaluate({m: 'Account 001\nAmount 900' for m in 'abc'})
        rows = passages(result)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['status'] == 'corroborated' for r in rows))
        self.assertEqual(rows[0]['supporting_family_count'], 3)
        self.assertEqual(rows[0]['other_family_support_count'], 2)
        self.assertEqual((rows[1]['token_start'], rows[1]['token_end']), (2, 4))

    def test_wrong_number_is_conflict_even_when_majority_threshold_met(self):
        result = evaluate({'a': 'Amount 900', 'b': 'Amount 900', 'c': 'Amount 900', 'd': 'Amount 100'})
        row = passages(result)[0]
        self.assertEqual(row['supporting_family_count'], 3)
        self.assertEqual(row['required_family_count'], 2)
        self.assertEqual(row['status'], 'corroborated')
        self.assertEqual(row['uncertainty_reasons'], [])
        self.assertEqual(row['flags'], ['conflicting_readings'])
        self.assertEqual(row['family_evidence']['d']['members']['d']['peer_text'], 'amount 100')
        self.assertEqual(passages(result, 'd')[0]['supporting_family_count'], 1)

    def test_related_variants_do_not_multiply_votes(self):
        result = evaluate({m: 'same passage' for m in 'abcd'},
                          families={'a': ['A'], 'b': ['A'], 'c': ['B'], 'd': ['C']})
        self.assertEqual(passages(result)[0]['supporting_family_count'], 3)
        self.assertEqual(passages(result)[0]['total_family_count'], 3)

    def test_family_disagreement_abstains_and_is_flagged(self):
        result = evaluate({'a': 'Amount 900', 'b': 'Amount 100', 'c': 'Amount 900', 'd': 'Amount 900'},
                          families={'a': ['A'], 'b': ['A'], 'c': ['B'], 'd': ['C']})
        row = passages(result)[0]
        self.assertEqual(row['supporting_families'], ['B', 'C'])
        self.assertEqual(row['conflicting_families'], ['A'])

    def test_missing_family_member_cannot_be_hidden(self):
        result = evaluate({'a': 'Amount 900', 'b': '', 'c': 'Amount 900', 'd': 'Amount 900'},
                          families={'a': ['A'], 'b': ['A'], 'c': ['B'], 'd': ['C']})
        self.assertEqual(passages(result)[0]['family_evidence']['A']['state'], 'conflict')

    def test_failed_and_truncated_outputs_cannot_vote_or_shrink_denominator(self):
        result = evaluate({m: 'Amount 900' for m in 'abcdef'},
                          statuses={'e': 'failed', 'f': 'truncated'})
        row = passages(result)[0]
        self.assertEqual((row['supporting_family_count'], row['total_family_count'], row['required_family_count']),
                         (4, 6, 2))
        self.assertEqual(row['unavailable_families'], ['e', 'f'])
        self.assertEqual(row['status'], 'corroborated')
        self.assertIn('incomplete_source', passages(result, 'f')[0]['flags'])
        self.assertEqual(passages(result, 'f')[0]['status'], 'corroborated')

    def test_partially_available_family_does_not_vote(self):
        result = evaluate({m: 'same passage' for m in 'abcd'},
                          families={'a': ['A'], 'b': ['A'], 'c': ['B'], 'd': ['C']},
                          statuses={'b': 'failed'})
        self.assertEqual(passages(result)[0]['unavailable_families'], ['A'])
        self.assertEqual(passages(result)[0]['supporting_family_count'], 2)

    def test_all_empty_outputs_are_uncertain_not_verified_blank(self):
        result = evaluate({m: '' for m in 'abc'})
        self.assertEqual(result['passages'], [])
        self.assertEqual(result['summary']['empty_page_outputs'], 3)
        self.assertTrue(all(p['status'] == 'uncertain' for p in result['pages']))
        self.assertEqual(result['pages'][0]['uncertainty_reasons'], ['no_text_returned'])

    def test_lone_text_is_retained_and_flagged(self):
        result = evaluate({'a': 'rare faint text', 'b': '', 'c': ''})
        row = passages(result)[0]
        self.assertEqual(row['supporting_family_count'], 1)
        self.assertEqual(row['missing_families'], ['b', 'c'])
        self.assertEqual(row['status'], 'uncertain')

    def test_formatting_and_line_wraps_do_not_prevent_support(self):
        result = evaluate({'a': 'Total amount 900', 'b': 'total\namount 900', 'c': 'TOTAL AMOUNT 900'})
        self.assertTrue(all(p['status'] == 'corroborated' for p in result['passages']))

    def test_inserted_text_is_not_hidden_or_credited_as_same_passage(self):
        result = evaluate({'a': 'total amount 900', 'b': 'total extra amount 900', 'c': 'total amount 900'})
        row = passages(result)[0]
        self.assertEqual(row['conflicting_families'], ['b'])
        self.assertEqual(row['status'], 'corroborated')
        self.assertEqual(row['flags'], ['conflicting_readings'])
        self.assertEqual(passages(result, 'b')[0]['status'], 'uncertain')

    def test_repeated_passage_without_unique_context_is_ambiguous(self):
        repeated = ('same ' * 50).strip()
        result = evaluate({m: repeated for m in 'abc'}, passage_tokens=3)
        row = passages(result)[5]
        self.assertEqual(row['supporting_family_count'], 1)
        self.assertEqual(row['ambiguous_families'], ['b', 'c'])
        self.assertIn('ambiguous_alignment', row['flags'])

    def test_repeated_words_with_unique_surrounding_context_can_align(self):
        text = 'first account\nAmount 900\nsecond account\nAmount 900\nend'
        result = evaluate({m: text for m in 'abc'})
        self.assertTrue(all(p['status'] == 'corroborated' for p in result['passages']))

    def test_one_peer_occurrence_cannot_support_two_source_occurrences(self):
        result = evaluate({'a': 'Amount 900\nAmount 900', 'b': 'Amount 900', 'c': 'Amount 900'})
        self.assertFalse(any(p['status'] == 'corroborated' for p in passages(result)))

    def test_candidates_and_hybrids_never_add_panel_votes(self):
        texts = {m: 'same passage' for m in 'abc'}
        before = evaluate(texts)
        texts['d'] = 'same passage'
        families = {m: [m] for m in 'abc'}
        families['d'] = ['a', 'b']
        after = evaluate(texts, families=families, panel=list('abc'))
        self.assertEqual(before['passages'], [p for p in after['passages'] if p['method'] != 'd'])
        self.assertEqual(passages(after, 'd')[0]['other_family_support_count'], 1)
        with self.assertRaisesRegex(ValueError, 'Hybrid'):
            evaluate(texts, families=families)

    def test_page_identity_prevents_cross_page_support(self):
        data = {'doc': {m: {'pages': ['first page', 'second page']} for m in 'abc'}}
        data['doc']['c']['pages'].reverse()
        result = corroborate_saved(data, {m: [m] for m in 'abc'}, list('abc'))
        self.assertTrue(all(p['status'] == 'corroborated' for p in result['passages'] if p['method'] in 'ab'))
        self.assertTrue(all(p['status'] == 'uncertain' for p in result['passages'] if p['method'] == 'c'))

    def test_unaligned_or_malformed_inputs_rejected(self):
        base = {'doc': {m: {'pages': ['text']} for m in 'abc'}}
        for change in ({'pages': None}, {'pages': ['one', 'two']}, {'statuses': []},
                       {'statuses': ['unknown']}, {'pages': [None]}):
            data = deepcopy(base)
            data['doc']['a'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                corroborate_saved(data, {m: [m] for m in 'abc'}, list('abc'))

    def test_invalid_thresholds_rejected(self):
        for settings in ({'min_families': 1}, {'min_families': True}, {'min_fraction': float('nan')},
                         {'min_fraction': -0.1}, {'min_fraction': 1.1}, {'passage_tokens': 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                corroboration_policy(**settings)

    def test_configurable_threshold_and_csv_review_fields(self):
        result = evaluate({'a': 'same', 'b': 'same', 'c': ''}, min_families=2, min_fraction=0.5)
        row = passages(result)[0]
        self.assertEqual(row['status'], 'corroborated')
        csv_row = next(passage_csv_rows(result))
        self.assertEqual(csv_row['supporting_families'], 'a; b')
        self.assertNotIn('family_evidence', csv_row)
        self.assertEqual(csv_row['text'], 'same')

    def test_default_accepts_two_of_six_but_not_one(self):
        result = evaluate({'a': 'same passage', 'b': 'same passage', 'c': 'same passage',
                           'd': '', 'e': '', 'f': ''})
        row = passages(result)[0]
        self.assertEqual(row['required_family_count'], 2)
        self.assertEqual(row['status'], 'corroborated')
        result = evaluate({'a': 'same passage', 'b': 'same passage', 'c': '',
                           'd': '', 'e': '', 'f': ''})
        self.assertEqual(passages(result)[0]['status'], 'corroborated')
        result = evaluate({'a': 'same passage', 'b': '', 'c': '', 'd': '', 'e': '', 'f': ''})
        self.assertEqual(passages(result)[0]['uncertainty_reasons'], ['insufficient_family_support'])

    def test_two_family_default_is_fixed_even_for_larger_panels(self):
        result = evaluate({m: 'same' if m in 'ab' else '' for m in 'abcdefghij'})
        self.assertEqual(passages(result)[0]['required_family_count'], 2)
        self.assertEqual(passages(result)[0]['status'], 'corroborated')

    def test_fraction_can_restore_previous_three_of_six_policy(self):
        result = evaluate({m: 'same' if m in 'ab' else '' for m in 'abcdef'},
                          min_families=3, min_fraction=0.5)
        self.assertEqual(passages(result)[0]['required_family_count'], 3)
        self.assertEqual(passages(result)[0]['status'], 'uncertain')

    def test_three_vs_three_corroborates_both_readings_and_preserves_conflicts(self):
        result = evaluate({m: 'Amount 900' if m in 'abc' else 'Amount 100' for m in 'abcdef'})
        for row in result['passages']:
            self.assertEqual(row['status'], 'corroborated')
            self.assertEqual(row['supporting_family_count'], 3)
            self.assertEqual(len(row['conflicting_families']), 3)
            self.assertEqual(row['uncertainty_reasons'], [])
            self.assertEqual(row['flags'], ['conflicting_readings'])
        self.assertEqual(result['summary']['corroborated_with_flags'], 6)
        self.assertTrue(all(p['flagged_passages'] == 1 for p in result['pages']))
        self.assertEqual(next(passage_csv_rows(result))['flags'], 'conflicting_readings')

    def test_ambiguous_peer_does_not_veto_three_supporting_families(self):
        result = evaluate({'a': 'same phrase', 'b': 'same phrase', 'c': 'same phrase',
                           'd': 'same phrase same phrase', 'e': '', 'f': ''})
        row = passages(result)[0]
        self.assertEqual(row['status'], 'corroborated')
        self.assertEqual(row['supporting_family_count'], 3)
        self.assertEqual(row['ambiguous_families'], ['d'])
        self.assertEqual(row['flags'], ['ambiguous_alignment'])
        self.assertEqual(row['uncertainty_reasons'], [])


if __name__ == '__main__':
    unittest.main()
