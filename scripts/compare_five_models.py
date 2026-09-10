"""Compare five saved OCR methods using whole-document text agreement."""
import csv
import hashlib
import json
import statistics

from cross_model_eval import evaluate
from evaluate_outputs import ALGORITHM_VERSION, DOCS, load
from ocr_common import ROOT, write_json


def methodology():
    return f"""## Evaluation methodology

### Scope and inputs

Algorithm version: **{ALGORITHM_VERSION}**. The evaluation compares saved OCR text from
Mistral, openai-6, fable-5-1, MinerU and Insavlo for three source documents:

| Document | Source pages | Weight in final score |
|---|---:|---:|
| CSA agreement | 14 | 1/3 |
| Board resolution | 4 | 1/3 |
| Trade documents | 30 | 1/3 |

The comparison unit is a **whole document**. Existing pages are concatenated in
their saved order. Insavlo's Markdown does not provide consistent physical page
boundaries for all documents, so this run does not calculate an average of
48 aligned page scores. Each method contributes one text per document.

Mistral text is extracted from response Markdown with referenced table contents
inserted. MinerU text comes from saved content blocks. openai-6 uses saved page
Markdown; fable-5-1 uses raw `full_page_text`, excluding separately reviewed
fields. Insavlo uses its saved document Markdown. The repository loader rejects
missing or ambiguous matching runs. Artifact paths, available hashes and source
verification status are recorded in [results.json](results.json).

### Text preparation

1. Decode Markdown/HTML formatting once; preserve plain-text inputs. Keep text
   inside formatting, preserve unknown placeholders such as `<REDACTED>`, and
   remove image links. Inline emphasis does not insert spaces within words;
   block elements and table cells introduce separators.
2. Apply Unicode NFKC normalization, remove soft hyphens and collapse whitespace.
   Join a word split by a line-break hyphen only if another output in the same
   comparison contains the joined word. In this run, that comparison spans the
   whole document, so corroboration can come from anywhere in a peer document.
3. Tokenize case-insensitively. Retain signed numbers, decimal/date separators,
   percentages, leading zeroes and accounting parentheses. For example,
   `(100.00)` and `100.00` differ; parentheses are preserved without assuming
   every parenthesized number is negative. Most other punctuation is ignored.

Repeated token occurrences are retained. This is not semantic matching: spelling
variants remain disagreements. Explicit descriptions such as “A Signature Here”
remain text tokens if present in an output.

### Primary pairwise score

For two texts A and B, let `count_A(t)` and `count_B(t)` be the occurrence counts
of token t. Define:

```text
shared(A, B) = sum over tokens t of min(count_A(t), count_B(t))
overlap(A, B) = 2 × shared(A, B) / (token_count(A) + token_count(B))
```

This is a multiset Dice coefficient between 0 and 1. For example, `a a b`
versus `a b` has two shared occurrences and scores `2 × 2 / 5 = 0.8`.
Reordering tokens does not change the score. Missing text, extra repetitions
and changed numeric tokens can lower it. No sequence or structure weight is
added to the primary score.

### Aggregation and ranking

With five methods, there are ten pairs per document and thirty pairs in this run.
Each method is compared with its four peers:

```text
document_score(method, document) = 100 × mean(overlap with each of four peers)
final_score(method) = mean(document_score across the three documents)
```

All peers receive equal weight, and each document receives one third of the
final weight, regardless of length. Sort final scores descending. Numerical ties
within an absolute tolerance of `1e-12` on the 0–1 scale share a competition rank;
display rounding to two decimals does not create a tie. The overall agreement
shown above is the mean of the five final method scores, not an accuracy rate.

### Separate diagnostics and empty text

- **Sequence similarity:** mean of forward and reverse token-level
  `difflib.SequenceMatcher` ratios with `autojunk=False`. It is order-sensitive,
  is not a Levenshtein error rate, and does not affect the ranking.
- **Numeric agreement:** the same occurrence-count overlap formula restricted
  to numeric strings. Missing repeated amounts count as disagreements. If neither
  text has numbers, the diagnostic is `null`, not perfect agreement. Numeric
  tokens already participate in the primary score; this diagnostic adds no weight.
- **Differences:** token alignment passages and unmatched numeric occurrence
  counts are saved in [document_differences.json](document_differences.json).
  A moved passage can appear as a deletion/insertion even when content overlap is high.
- **Empty text:** units where every method lacks scorable tokens are excluded.
  An empty/nonempty pair scores zero; an empty/empty pair also scores zero when
  other methods contain text. An entirely empty evaluation raises an error.

### Interpretation and reproduction

This measures **text consensus, not OCR correctness, completeness against the
source, or model accuracy**. Correct unique readings can be penalized and shared
errors rewarded. Identical token counts can conceal wrong word associations;
whole-document comparison can also hide movement across pages. Related OCR
pipelines do not provide independent votes. Model identity and source preparation
are only partly verified, as described above. No layout, speed, cost or visual
quality score is included, and no statistical significance is claimed from
three documents. Prior visual spot checks are not ground truth used in scoring.

Reproduce from the repository root without rerunning OCR:

```bash
conda run --no-capture-output -n myenv3.13 python scripts/compare_five_models.py
```

For future aligned single-page evaluation, `score_page(paths)` returns one
0–100 consensus score; `score_models(paths)` returns per-method scores in input
order. Keep the same methods and ordering across pages before averaging.
`score_page_details(paths)` additionally computes the separate diagnostics.
"""


def main():
    data, provenance = load(normalize=False)
    output = ROOT / 'output' / 'five_model_comparison'
    insavlo_paths = {
        'csa': ROOT / 'output/insavlo' / (DOCS['csa'] + '.md'),
        'board': ROOT / 'output/insavlo/board_resolution.md',
        'trade': ROOT / 'output/insavlo/Trade-1118348-260625.001369.01.01.tif0.tiff.md',
    }
    source_page_counts = {}
    for document, methods in data.items():
        source_page_counts[document] = {m: len(record['pages']) for m, record in methods.items()}
        for record in methods.values():
            record['pages'] = ['\n\n'.join(record['pages'])]
        path = insavlo_paths[document]
        raw = path.read_bytes()
        methods['insavlo'] = {'pages': [raw.decode('utf-8-sig')]}
        provenance.append({'document': document, 'engine_label': 'insavlo',
                           'artifacts': [str(path)], 'sha256': hashlib.sha256(raw).hexdigest(),
                           'source_hash_verified': None,
                           'metadata': 'Markdown only; model provenance and physical page boundaries unavailable.'})
    result, pairs, volumes, normalized = evaluate(data, overlap_weight=1.0, weighting='document')
    # The evaluator uses one text unit per record: here each unit is a DOCUMENT.
    for row in pairs + volumes:
        row.pop('page', None)
        row['unit'] = 'whole_document'
    for row in result['pairwise_summary']:
        row['documents'] = row.pop('pages')
        row['documents_with_numbers'] = row.pop('pages_with_numbers')
        for key in list(row):
            if key.startswith('mean_page_'):
                row[key.replace('mean_page_', 'mean_document_')] = row.pop(key)
    result['settings']['comparison_unit'] = 'whole_document'
    result['settings']['aggregation'] = 'Equal weight for each of three documents; not a page average.'
    result['source_page_counts'] = source_page_counts
    result['excluded_documents'] = result.pop('excluded_pages')
    result['provenance'] = provenance
    result['overall_agreement_score'] = statistics.mean(r['consensus_score'] for r in result['ranking'])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'results.json', result)
    write_json(output / 'document_differences.json', pairs)
    write_json(output / 'text_volume.json', volumes)
    write_json(output / 'normalized_documents.json', {
        d: {m: r['pages'][0] for m, r in methods.items()} for d, methods in normalized.items()})
    with (output / 'ranking.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['rank', 'model', 'consensus_score'])
        writer.writeheader()
        writer.writerows(result['ranking'])
    lines = ['# Five-method text consensus comparison', '',
             'Three complete documents; each document has equal weight. Insavlo lacks consistent physical page boundaries, so this is a whole-document comparison, not a page-average comparison. All methods use the same document scope.', '',
             f'Algorithm version {ALGORITHM_VERSION}: primary agreement is token-count overlap, independent of reading order. Each method is scored against its four peers, then averaged across documents. Formatting and case are ignored; signed numbers, decimal separators, accounting parentheses and percentages are retained. Sequence similarity and numeric agreement are separate diagnostics and do not contribute additional weight to the ranking. Numeric agreement counts repeated occurrences. Numeric tokens are included in token overlap. No structure or layout score is used. A reordered text can receive 100 even when word associations differ; this is a content-consensus measure.', '',
             '**Consensus is not accuracy.** Shared errors can increase scores. The openai-6 and fable-5-1 artifacts describe Tesseract with visual review; their folder labels do not verify the named models. Insavlo has no model provenance metadata. No OCR was rerun.', '',
             '| Rank | Method | Mean consensus / 100 | CSA | Board | Trade |',
             '|---:|---|---:|---:|---:|---:|']
    for row in result['ranking']:
        per_doc = [next(r['consensus_score'] for r in result['per_document_rankings'][d]
                        if r['model'] == row['model']) for d in ('csa', 'board', 'trade')]
        lines.append(f"| {row['rank']} | {row['model']} | {row['consensus_score']:.2f} | " +
                     ' | '.join(f'{score:.2f}' for score in per_doc) + ' |')
    lines += ['', f"Overall agreement across all methods and documents: {result['overall_agreement_score']:.2f}/100.", '',
              'These scores are not directly comparable to the earlier four-method page-based scores: both the peer set and comparison unit changed.', '',
              'Files: results.json includes per-document rankings, pairwise summaries and provenance; document_differences.json contains text and numeric disagreements; ranking.csv contains the overall ranking.']
    lines += ['', methodology()]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result['ranking'], indent=2))
    print(output / 'report.md')


if __name__ == '__main__':
    main()
