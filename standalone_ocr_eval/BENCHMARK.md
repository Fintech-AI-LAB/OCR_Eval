# Critical-field OCR benchmark

The reference benchmark complements the existing consensus evaluator. Document
profiles define what matters; independently reviewed annotations define what is
actually on each source; predictions are scored against those fixed annotations.
An OCR method cannot remove a field from evaluation by missing its label.

The implementation uses only Python's standard library. In this repository,
run all Python commands in Conda `myenv3.13`.

## Run the synthetic demonstration

From the repository root:

```bash
conda run --no-capture-output -n myenv3.13 python scripts/benchmark_ocr.py benchmark \
  --profiles standalone_ocr_eval/examples/benchmark/profiles.json \
  --references standalone_ocr_eval/examples/benchmark/references.json \
  --predictions standalone_ocr_eval/examples/benchmark/predictions.json \
  --output output/critical_field_demo
```

The standalone equivalent, from this package directory after installation, is
`python -m ocr_eval benchmark` with the same options and adjusted paths.

The example includes perfect output, an incorrect amount, amounts exchanged
between transactions, a missing row, and an empty output. All values, geometry,
weights and methods are **synthetic examples**, not measured engine performance
or reviewed business requirements. Every result and report retains this status.

Outputs: `results.json`, `ranking.csv`, and `report.md`. The JSON contains
per-field/document/template diagnostics, relationship results, raw value pairs,
locations, extraction issues, source constraints, and input hashes.

## 1. Define and review a document profile

Use a separate profile for each document type or task. The broad repository
ontology remains a vocabulary source; membership alone does not establish
criticality. A profile contains competency questions, review status, and fields:

```json
{
  "schema_version": 1,
  "profiles": {
    "payment": {
      "review_status": "example",
      "competency_questions": ["What amount and currency apply to each payment?"],
      "record_key_labels": ["Transaction", "Record"],
      "fields": {
        "Amount": {
          "ontology_id": "concept:Amount",
          "aliases": ["Settlement Amount"],
          "value_type": "amount",
          "critical": true,
          "required": true,
          "weight": 5,
          "rationale": "Illustrative cost: an incorrect value changes the obligation.",
          "requires": ["Currency"]
        },
        "Currency": {
          "value_type": "currency",
          "critical": true,
          "required": true,
          "weight": 2,
          "rationale": "Illustrative cost: currency determines the unit of settlement.",
          "constraints": {"enum": ["USD", "EUR", "SGD"]}
        }
      }
    }
  }
}
```

Have a domain reviewer set the costs, applicability, aliases, and relationships;
then use `review_status: "reviewed"` and `reviewed_by`. Every field needs a
positive finite weight, a rationale, a value type, and an explicit critical flag.
Noncritical fields still receive value and detection metrics, but do not enter
the primary score. Ambiguous aliases are rejected; use scoped names or different
profiles instead. `ontology_id` is optional descriptive provenance, not an
automatically validated link into an external ontology.

Types are `text`, `identifier`, `amount`, `number`, `date`, and `currency`.
`comparison: "typed"` (default) uses the existing conservative normalizer:
identifiers preserve leading zeros and case except standard IBAN/BBAN/BIC/ISIN/
LEI/UETR names; those normalize spaces and case. Text normalizes Unicode,
whitespace and case. Currency codes normalize case. Numbers use exact decimal
values with dot decimals and grouped thousands commas, retaining sign,
percent/per-mille units, and any embedded currency. Accounting parentheses are
negative amounts. Ambiguous numeric dates require `date_order: "dmy"` or `"mdy"`.
There is no exchange-rate conversion or fuzzy numeric matching.

`comparison: "literal"` requires exact raw-string equality, including case and
spacing. Use it when the source's literal text matters or cannot satisfy the
typed policy. The separate literal accuracy and character-error metric always
compare raw strings. Nonempty source values that cannot be normalized under a
typed profile are rejected as invalid annotations; correct the annotation or
explicitly choose literal comparison. Values are never repaired.

`required` checks presence within applicable record scopes. `requires` describes
direct associations to other fields in the same record. Constraints support raw
full-match `pattern`, normalized `enum`, and numeric `minimum`/`maximum`. Numeric
bounds apply to the numeric component in its written unit. Constraints report
plausibility separately from accuracy; a source can itself violate a constraint.

## 2. Annotate source documents independently

Create a manifest with document IDs, profile IDs, template IDs, source paths,
record IDs, and a `train`, `validation`, or `test` split. Source paths are relative
to the manifest file. Example: `examples/benchmark/annotation-manifest.json`.

```bash
conda run --no-capture-output -n myenv3.13 python scripts/benchmark_ocr.py init-reference \
  standalone_ocr_eval/examples/benchmark/annotation-manifest.json \
  --profiles standalone_ocr_eval/examples/benchmark/profiles.json \
  --output output/annotations/draft.json
```

This hashes the actual source files and creates **unannotated** slots. It refuses
to overwrite an existing annotation file. Draft references cannot be scored.
Creating a template does not perform OCR or establish any ground truth.

For each record, annotate every profile field with one status:

| Status | Annotation | Evaluation |
|---|---|---|
| `present` | Literal value, page, box | Fixed value/localization denominator |
| `absent` | No value | A prediction is an extra field |
| `not_applicable` | No value | No expected value or requiredness check; a prediction is extra |
| `unreadable` | Page and box, no value | Predictions ignored and counted; no invented truth |

Use normalized bounding boxes `[left, top, right, bottom]` in `[0, 1]`, with
positive area and one-based page numbers. Record IDs are stable, case-sensitive
identities; each field occurs at most once per record. Give separate line items
separate record IDs. Fields repeated within a record need a finer record schema,
not occurrence-based matching.

Minimal reference document (inside a schema-version-1 `documents` mapping):

```json
{
  "profile": "payment",
  "template_id": "supplier-layout-01",
  "split": "test",
  "reviewed_by": "Human reviewer identifier",
  "source": {"path": "source.pdf", "sha256": "64 lowercase hex characters"},
  "records": [{"id": "T1", "fields": {
    "Amount": {"status": "present", "value": "100.00", "page": 1, "bbox": [0.1, 0.2, 0.3, 0.25]},
    "Currency": {"status": "present", "value": "USD", "page": 1, "bbox": [0.3, 0.2, 0.4, 0.25]}
  }}],
  "regions": [{"id": "payment-row", "text": "100.00 USD", "page": 1, "bbox": [0.1, 0.2, 0.4, 0.25]}]
}
```

The source hash placeholder above must be replaced with an actual SHA-256. Set
the top-level `annotation_status` to `verified` only after image review and add
each document's reviewer. A verified benchmark also requires reviewed profiles.
Verification is an annotator declaration; the evaluator does not automatically
prove image correctness or reread source files during scoring.

The optional `regions` list defines **fixed source regions for the OCR-only
track**. Transcribe the exact region and run each OCR method on the same crop,
or map each method's spatial output to that fixed region using an independently
defined procedure. The evaluator consumes these transcripts; it does not crop
images, call OCR services, or guess region text from model-selected labels.

Keep templates entirely within one split: cross-split template reuse is rejected.
Do not tune aliases or costs on the test annotations. Use independently annotated
documents beyond the repository's three-document pilot for reliable conclusions.

## 3. Supply predictions without access to reference values

```json
{
  "schema_version": 1,
  "methods": {
    "engine-a": {
      "family": "actual-engine-family",
      "provenance": {"model": "exact model/revision", "settings": {}},
      "documents": {
        "document-id": {
          "fields": [
            {"record": "T1", "field": "Amount", "value": "100", "page": 1, "bbox": [0.1, 0.2, 0.3, 0.25]},
            {"record": "T1", "field": "Currency", "value": "USD", "page": 1, "bbox": [0.3, 0.2, 0.4, 0.25]}
          ],
          "regions": {"payment-row": "100 USD"}
        }
      }
    }
  }
}
```

Supply every reference document for every method, including documents outside
the selected scoring split; use `fields: []` for failed or empty outputs. Missing
documents raise an error rather than silently reducing the sample. Unknown field
IDs and unexpected record IDs count as extras; duplicate predictions count too.
Fields require raw string values. Prediction boxes are optional, but omitted
boxes receive no localization credit. Optional `source_sha256` must match the
reference source when supplied, and results record whether it was checked.

For existing aligned-page JSON, a deterministic baseline extractor is available:

```bash
conda run --no-capture-output -n myenv3.13 python scripts/benchmark_ocr.py extract-profile \
  standalone_ocr_eval/examples/benchmark/aligned.json \
  --profiles standalone_ocr_eval/examples/benchmark/profiles.json \
  --document-profiles standalone_ocr_eval/examples/benchmark/document-profiles.json \
  --families standalone_ocr_eval/examples/benchmark/families.json \
  --output output/profile_predictions.json
```

The document/profile file maps IDs directly, e.g.
`{"synthetic-trade": "trade_confirmation"}`. This extractor never reads references.
It handles explicit whole-line `Label: value` / `Label = value`, adjacent
label/value lines, two-column key/value tables, and Markdown/HTML/TSV tables with
a declared record-key column. It reads complete table rows beyond a header window.
Unkeyed tables are reported as issues; they are not assigned invented record IDs.
Unkeyed label/value pairs use record ID `document`. It does not infer fuzzy labels,
recover arbitrary prose or spatial layout, or combine multiple inline pairs.
Choose an external extractor for those cases and submit the same prediction schema.

Extraction produces no boxes or fixed-region transcripts. Those metrics therefore
receive zero credit/deletion errors when corresponding references exist. Supply
them separately for the OCR-only/location tracks. The extractor hashes its profile
and benchmark scoring rejects a different profile, preventing unnoticed drift.

## Metrics and matching policy

Match `(document, record, canonical field)` one-to-one. Among duplicate candidates,
prefer a correct typed value, then a literal match, then greater box overlap;
all remaining duplicates are extras. Reordering records has no effect. Moving a
value to a different record cannot earn credit by matching elsewhere in the page.

The primary score is:

`100 × sum(weight × correct critical field) / sum(reference critical-field weights)`

A critical field is correct when its value and all its **direct, readable,
present** `requires` targets in the same record are correct. Relationships whose
reference target is absent/unreadable are not scored; coverage and source
constraint issues remain visible. Noncritical associated values can therefore
affect a critical field. All costs and relationships must be fixed before testing.

By default, compute this score per document and average documents equally.
`--weighting field` pools the weighted field counts. The primary score measures
reference recovery and does not directly penalize extra outputs; always inspect
value precision and document pass rate alongside it. Documents without readable
critical fields are listed explicitly and excluded from this score.

| Metric | Definition |
|---|---|
| Field precision/recall | Nonblank predictions matched to the correct record/field identity, regardless of value |
| Value precision/recall/F1 | Correct typed/literal values at the correct record/field identity |
| Literal value accuracy | Exact raw-string equality over reference-present fields |
| Critical-field accuracy | Unweighted correctness of critical values plus declared direct relationships |
| Relationship accuracy | Both endpoint values correct within the same reference record |
| Record accuracy | All readable critical fields correct, with no extra critical fields in that record |
| Document pass rate | All critical fields correct, with no extra critical fields, their declared association targets, or unknown fields anywhere in the document |
| Localization recall | Reference-present fields with matching identity and box IoU at least `--iou-threshold` (default 0.5) on the same page |
| Field CER | Character edit distance at fixed field identities, including missing-value deletions and extra-value insertions |
| Critical-region CER | Character edit distance for independently specified reference regions |

Literal transcription and relationship-sensitive critical accuracy are separate.
Documents/records with unreadable critical fields or required association targets are ineligible for their pass
rates and report the reason. Empty outputs receive zero recovery, not exclusion.
Unreadable predictions are counted as ignored; absent/not-applicable predictions
are extras. An empty prediction value is a miss plus an invalid extra prediction.
Source constraint violations do not lower OCR accuracy when the source was read
correctly. Requiredness checks skip annotated unreadable/not-applicable fields.

Ratios are 0–1; ranking scores are 0–100. CER can exceed 1 because of insertions.
Undefined ratios are JSON `null`. Missing fixed-region transcripts are empty
strings for scoring; unknown region IDs are errors. Annotation input schemas,
finite costs, source hashes, duplicate JSON keys and bounding boxes are validated.

## Controlled comparisons and family sensitivity

Run extraction variants against the same reviewed profile/reference corpus:

```bash
conda run --no-capture-output -n myenv3.13 python scripts/benchmark_ocr.py compare-variants \
  --profiles profiles.json --references references.json \
  --variant dictionary=predictions_dictionary.json \
  --variant ontology=predictions_ontology.json \
  --output output/benchmark_variants
```

The first variant is the baseline. `variants.json` contains complete comparable
results and score differences for the same methods. Generate each variant's
predictions independently; keep reference fields, costs, normalization and scoring
fixed. This evaluates the extraction/localization contribution of the ontology.
Changing the evaluation targets or costs is a different experiment and cannot be
interpreted as an extraction improvement. No extractor is trained by this command.

The regression suite includes missing rows/fields, changed digits, identifier
leading zeros, exchanged row values, empty output, duplicates, incorrect record
IDs and label aliases. Extend these controlled cases with actual scan degradation
and unusual templates before selecting an engine.

For existing consensus `page_differences.json`, provide actual engine families:

```bash
conda run --no-capture-output -n myenv3.13 python scripts/benchmark_ocr.py family-sensitivity \
  output/cross_model/page_differences.json --families engine_families.json \
  --output output/family_sensitivity.json
```

This excludes same-family peers, averages peers within each other family, and then
weights families equally. It also removes each family in turn. Page scores are
averaged within each document and documents equally. Complete peer matrices on
the supplied units are required; fewer than two remaining families yields `null`.
These are **consensus sensitivity diagnostics**, not accuracy or independent truth.
An omitted family mapping during extraction defaults to each method's own name;
replace those defaults with actual engine lineage before sensitivity analysis.

## Python API

```python
from ocr_eval import evaluate_benchmark, compare_benchmark_variants, load_profiles
from ocr_eval.profiles import load_json

profiles = load_profiles("profiles.json")
references = load_json("references.json")
predictions = load_json("predictions.json")
result = evaluate_benchmark(profiles, references, predictions, split="test")
```

Existing `score_page`, `score_models`, `evaluate`, and their defaults are unchanged.
The new benchmark version is independent of the consensus algorithm version.

## Literature informing the design

- Noy and McGuinness (2001), [Ontology Development 101](https://protege.stanford.edu/publications/ontology_development/ontology101-noy-mcguinness.html): competency questions and task-specific scope motivate document profiles.
- Elkan (2001), [The Foundations of Cost-Sensitive Learning](https://cseweb.ucsd.edu/~elkan/rescale.pdf): differentiated error costs motivate reviewed field weights. The OCR score here is an adaptation, not a metric validated in that paper.
- Jaume et al. (2019), [FUNSD](https://arxiv.org/abs/1905.13538): separate recognition, entity labeling and linking tasks motivate separate evaluation tracks.
- Šimsa et al. (2023), [DocILE](https://arxiv.org/abs/2302.05658): field type/location and line-item grouping motivate record-aware annotations and template-based evaluation.
- van Strien et al. (2020), [Assessing the Impact of OCR Quality on Downstream NLP Tasks](https://www.repository.cam.ac.uk/items/ed38e0dc-410a-4431-bbef-f96ff1c0c3db): downstream evaluation motivates complete-record/document success metrics.
- [W3C SHACL](https://www.w3.org/TR/shacl/): datatype and cardinality validation inform the lightweight constraint checks. This implementation is JSON-based and is not a SHACL processor.
- Bach et al. (2017), [Learning the Structure of Generative Models without Labeled Data](https://proceedings.mlr.press/v70/bach17a.html): dependence between noisy sources motivates family sensitivity. This implementation does not fit the paper's probabilistic model.

Learned spatial extraction, fuzzy label recovery, automatic image annotation,
probabilistic consensus calibration and statistical confidence intervals are not
implemented. The supplied framework supports independently produced predictions
and exposes the evidence needed to validate those extensions.
