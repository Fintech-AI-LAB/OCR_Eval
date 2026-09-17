# Cross-model text evaluation

Documents: 1; methods: 3; excluded all-empty pages: 0.

**Scores measure consensus, not correctness.** No ground-truth model or structure score is used.

Pair score = 1.00 × ontology-weighted token overlap + 0.00 × symmetric sequence similarity. Average across peers, then use equal document weighting. Scoring ignores formatting and case, but preserves signed numbers, numeric separators, percentages and leading zeroes. Numeric agreement is reported separately. Line hyphens are joined only when another output corroborates the joined word.

Ontology concept/field mentions receive weight 3; other tokens receive weight 1. Matching is lexical and does not infer entity values. See results.json for ontology provenance.

| Rank | Saved model label | Consensus / 100 |
|---:|---|---:|
| 1 | 0 | 54.17 |
| 1 | 1 | 54.17 |
| 3 | 2 | 33.33 |

Empty text is not evidence of a correct blank page. All-empty pages are excluded; two empty methods receive zero agreement when another method has text. Ties share ranks.

Model labels are supplied by the input. Shared errors and related OCR pipelines can inflate agreement. The repository openai-6 and fable-5-1 artifacts identify Tesseract with visual review; their labels do not verify those named models. Fable raw page text is used, not separately reviewed fields.

See results.json for per-document rankings, pairwise summaries and provenance; page_differences.json for differing passages and numeric strings. No original OCR files were modified and no inference calls were made.
