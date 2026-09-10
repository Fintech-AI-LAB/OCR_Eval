"""Reproduce Insavlo score decomposition and a controlled reading-order diagnostic."""
import collections
import json
import re
import statistics
from pathlib import Path
from evaluate_outputs import compare_page, normalize_texts, tokens
from ocr_common import ROOT, write_json


def main():
    # This investigation explains the historical version-2 reading-order penalty.
    base = ROOT / 'output/five_model_comparison/algorithm_v2_snapshot'
    pairs = json.loads((base / 'document_differences.json').read_text())
    documents = json.loads((base / 'normalized_documents.json').read_text())
    breakdown = []
    for doc in documents:
        rows = [r for r in pairs if r['document'] == doc and 'insavlo' in (r['left'], r['right'])]
        overlap = 100 * statistics.mean(r['token_overlap'] for r in rows)
        sequence = 100 * statistics.mean(r['sequence_similarity'] for r in rows)
        breakdown.append({'document': doc, 'token_overlap': overlap, 'sequence_similarity': sequence,
                          'combined': (overlap + sequence) / 2,
                          'reduction_from_sequence_component': (overlap - sequence) / 2})
    # Reorder the same signatory words, without changing token counts, in memory.
    raw = (ROOT / 'output/insavlo/board_resolution.md').read_text()
    pattern = r"(?m)^\| CATEGORY 'A' \| CATEGORY 'B' \|\n\|[^\n]+\|\n((?:\|[^\n]*\|\n)+)"
    def columns(match):
        rows = [line.strip().strip('|').split('|') for line in match[1].splitlines()]
        return "CATEGORY 'A'\n" + '\n'.join(r[0] for r in rows) + "\nCATEGORY 'B'\n" + '\n'.join(r[1] for r in rows) + '\n'
    reordered, count = re.subn(pattern, columns, raw)
    before, after = normalize_texts([raw, reordered])
    if count != 2 or collections.Counter(tokens(before)) != collections.Counter(tokens(after)):
        raise ValueError('Controlled table-order diagnostic changed token content or failed to find both tables.')
    diagnostic = []
    for peer, text in documents['board'].items():
        if peer == 'insavlo': continue
        a = compare_page('board', 1, 'insavlo', peer, before, text)
        b = compare_page('board', 1, 'insavlo', peer, after, text)
        diagnostic.append({'peer': peer, 'overlap_before': a['token_overlap'] * 100,
                           'overlap_after': b['token_overlap'] * 100,
                           'sequence_before': a['sequence_similarity'] * 100,
                           'sequence_after': b['sequence_similarity'] * 100,
                           'combined_before': 50 * (a['token_overlap'] + a['sequence_similarity']),
                           'combined_after': 50 * (b['token_overlap'] + b['sequence_similarity'])})
    result = {'algorithm_version': '2.0', 'breakdown': breakdown,
              'mean_overlap': statistics.mean(r['token_overlap'] for r in breakdown),
              'mean_sequence': statistics.mean(r['sequence_similarity'] for r in breakdown),
              'controlled_table_order_diagnostic': diagnostic,
              'diagnostic_changes_token_counts': False,
              'note': 'Diagnostic only; no production scores or original OCR files were modified.'}
    write_json(base / 'insavlo_investigation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
