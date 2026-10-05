# OCR evaluation

## Reference-based critical-field benchmark

Use the new [benchmark guide](standalone_ocr_eval/BENCHMARK.md) to define reviewed
document profiles, create image-annotation templates, and measure field accuracy
against fixed references. It reports missing/extra/wrong values, record and
currency associations, literal transcription, localization, constraints and
document success. Existing consensus commands remain unchanged.

Run the explicitly synthetic example (no OCR requests):

```bash
conda run --no-capture-output -n myenv3.13 python scripts/benchmark_ocr.py benchmark --profiles standalone_ocr_eval/examples/benchmark/profiles.json --references standalone_ocr_eval/examples/benchmark/references.json --predictions standalone_ocr_eval/examples/benchmark/predictions.json --output output/critical_field_demo
```

The CLI also provides `init-reference`, `extract-profile`, `compare-variants`,
and `family-sensitivity`. Real accuracy results require reviewed profiles and
independent image-verified references; existing peer-consensus outputs are not
ground truth. The example profiles and annotations are illustrative only.

## Standalone Python library

See [standalone_ocr_eval/README.md](standalone_ocr_eval/README.md) for installation, API, CLI, input schema, and optional ontology instructions.

## DeepSeek OCR 2 (local Apple Silicon)

```bash
bash scripts/setup_deepseek.sh
conda run --no-capture-output -n myenv3.13 python scripts/run_deepseek_ocr.py
```

Uses MLX-VLM 0.7.0 and the community BF16 conversion of DeepSeek OCR 2,
with a pinned model revision. Model weights download from Hugging Face;
document processing stays local. Requires an Apple Silicon Mac with Metal GPU
access. The model download is approximately 6.8 GB.

Dependencies live in `.cache/deepseek-ocr2-packages`, loaded only by this runner
using the existing `myenv3.13` Python. This avoids replacing the Transformers 4
packages used by MinerU. No separate Python environment is created.

Defaults process the three PDFs in `data/` (48 pages), render at 144 DPI, and
save to `output/deepseek-ocr2/`. Each run contains raw text and cleaned Markdown
in `response-NNNN.json`, combined `document.md`, and model/settings provenance
in `source.json`. Grounding markers are removed only from the Markdown;
raw model text is retained. Pages reaching the 8192-token cap are preserved with
`truncated: true` and a visible Markdown warning. The source metadata lists
`truncated_pages` and `status: needs_review`; the command exits nonzero after
processing the remaining pages. Use `--overwrite` to recompute cached responses,
or change decoding settings to create a separate run.

The runner supports `--dry-run`, `--input`, `--output`, `--limit`, `--overwrite`,
`--dpi`, `--max-tokens`, `--prompt`, `--no-cropping`, and
`--repetition-penalty`. It checkpoints completed pages for resume.
`--setup-only` downloads and loads the model without OCR.

The trade PDF hit the token cap on page 7 with default decoding and page 9
with repetition control alone. Full-page encoding passed those pages but still
reached the cap on page 10. The recorded full-document attempt uses:

```bash
conda run --no-capture-output -n myenv3.13 python scripts/run_deepseek_ocr.py --input data/Trade-1118348-260625.001369.01.01.tif0.pdf --no-cropping --repetition-penalty 1.1
```

Spot checks found transcription errors and omissions in dense trade tables.
These are model outputs for evaluation, not verified transcriptions. Raw model
responses are preserved without manual correction.

References: [MLX-VLM DeepSeek OCR 2](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr_2/README.md),
[model weights](https://huggingface.co/mlx-community/DeepSeek-OCR-2-bf16).

Both backends recursively process `data/` (PDF, JPEG, PNG, TIFF, WebP and BMP)
and save results under `output/<backend>/`. All Python commands use Conda
`myenv3.13`.

## Mistral (remote API)

```bash
conda activate myenv3.13
python -m pip install --no-user -r requirements-mistral.txt
export MISTRAL_API_KEY='your-api-key'
python scripts/run_mistral_ocr.py
```

PDFs are submitted directly; image frames, including multi-page TIFFs, are sent
as PNGs. This sends documents to Mistral and incurs API charges. No local GPU is
used. The API key is read from the environment and never saved in results.

## MinerU (local MPS or CPU)

```bash
bash scripts/setup_mineru.sh
conda run --no-capture-output -n myenv3.13 python scripts/run_mineru.py
```

Setup installs dependencies in the existing environment, verifies MPS, downloads
`opendatalab/MinerU2.5-Pro-2604-1.2B`, and runs a small inference smoke test.
Use `--skip-smoke-test` on setup to skip inference. It does not change Python.

The runner loads the model once, renders PDF pages at 144 DPI, and extracts every
page/image frame sequentially. `--dpi 200` adjusts PDF resolution. `--device mps`
or `--device cpu` forces a device; the default selects MPS when available.
Unsupported MPS operations may fall back to CPU. An MPS failure does not trigger
a full CPU retry. This uses MinerU's two-step extraction API, not its full PDF
pipeline; cross-page reconstruction is not included.

## Shared options and outputs

Both commands support:

```bash
python scripts/run_mistral_ocr.py --dry-run
python scripts/run_mineru.py --dry-run
python scripts/run_mistral_ocr.py --limit 1
python scripts/run_mineru.py --input data/example.pdf
```

`--input` accepts a file or directory; `--output` sets the backend output root.
`--limit` counts source files, not pages. `--dry-run` lists inputs without loading
models or calling APIs. `--overwrite` recomputes saved responses.

Each file produces `<relative-input-path>/<content-and-settings-hash>/` containing:

- `response-0001.json`, etc.: original backend responses, one per MinerU page or
  Mistral request (a PDF may have multiple pages in one response).
- `document.md`: combined Markdown, with HTML tables and embedded image data for
  Mistral where returned by the API.
- `source.json`: source path, model, settings, and response count.

Responses are checkpointed so rerunning resumes interrupted work. Source or
settings changes create a new result directory. The new shared cache format does
not reuse outputs produced by the earlier script version. Use `--overwrite` when
the model behind Mistral's `latest` alias changes. Failed files do not stop the
batch; the process exits nonzero if any fail. Oversized PDFs must be split before
Mistral submission. Avoid concurrent runs writing to the same backend output.

## Cross-model text evaluation

### Six saved methods, including Insavlo and DeepSeek

```bash
conda run --no-capture-output -n myenv3.13 python scripts/compare_six_models.py
```

Writes `output/six_model_comparison/` with rankings, per-document scores,
pairwise diagnostics, and source provenance. It compares Mistral, openai-6,
fable-5-1, MinerU, Insavlo, and the DeepSeek runs selected by
`output/deepseek-ocr2/manifest.json`. Because Insavlo lacks reliable page
boundaries, all methods are compared as whole documents with equal document
weight. DeepSeek's capped trade page 10 is included and explicitly flagged.
Scores measure text consensus, not accuracy or verified model identity.

### One page at a time

`score_page(file_paths)` accepts an array of OCR output paths for the same page
and returns a single float from 0 to 100. It averages all pairwise text agreement
scores. Use `score_models(file_paths)` for individual method scores in input
order, keeping that order consistent across pages when averaging for ranking.

```python
from statistics import mean
from scripts.score_page import score_page, score_models

# Each inner array contains different methods' outputs for ONE physical page.
pages = [
    ['outputs/mistral/page1.md', 'outputs/openai/page1.md', 'outputs/mineru/page1.json'],
    ['outputs/mistral/page2.md', 'outputs/openai/page2.md', 'outputs/mineru/page2.json'],
]
page_scores = [score_page(paths) for paths in pages]
average_score = mean(page_scores)

# Optional: average each method's score across pages, preserving input order.
method_scores = [score_models(paths) for paths in pages]
average_by_method = [mean(column) for column in zip(*method_scores, strict=True)]
```

Run your Python script in `myenv3.13`. To print just one page's numeric score:

```bash
conda run --no-capture-output -n myenv3.13 python scripts/score_page.py method_a.md method_b.md method_c.json
```

Supported files: UTF-8 text, Markdown, HTML, and single-page OCR JSON (Mistral
or OpenAI `pages`, Fable `full_page_text`, MinerU content-block lists, or a
`text`/`markdown` object). Split document outputs into single-page files first;
multi-page JSON is rejected. PDF/image files are not OCR text inputs.
At least two files are required; at least three are needed to distinguish
method rankings. All outputs without word/number tokens raise `ValueError`;
exclude such pages explicitly rather than treating them as perfect agreement.
The score measures consensus, not accuracy. Version 6 compares whole text regions
around ontology labels by default (`overlap_weight=1.0`). Changed regional text,
numbers and local word order reduce agreement; unmatched regions receive zero.
Pages without ontology anchors are excluded from batch scoring; single-page
scoring raises an error. Localization coverage is reported separately.
For diagnostics, `score_page_details(paths)` returns `score`, `model_scores`
(in input order), and `pairs` with separate token, sequence and numeric agreement.
Pair similarities are 0–1; the page and method scores are 0–100. Sequence and
numeric diagnostics have no additional weight in the default primary score.
Explicitly setting `overlap_weight=0.5` opts into a 50/50 region/sequence blend.

Version 3.1 preserves accounting parentheses as written: `(100.00)` differs
from `100.00` without assuming every parenthesized number is negative. Numeric
agreement now counts occurrences, so missing a repeated amount lowers it;
`unmatched_left_numeric_counts` and `unmatched_right_numeric_counts` identify
the missing or extra occurrences. Inline HTML emphasis does not split words,
while block elements and table cells remain separated.

The default `score_page()` and `score_models()` compare ontology regions, avoiding
global sequence alignment and full difference generation. Request
`score_page_details()` when those diagnostics are needed. Explicit sequence
weighting still requires alignment.

Normalization retains the version 2 fixes: signed numbers, decimal/date separators,
percentages and leading zeroes in scoring. Plain `.txt` files retain literal
angle-bracket text; Markdown and HTML formatting are decoded once before
idempotent plain-text normalization. Unknown tags such as `<REDACTED>` are
preserved. A word split by a line-break hyphen is joined only when another
output for that comparison contains the joined word. Soft hyphens are removed.
These rules reduce formatting penalties but do not establish which OCR is right.
The batch JSON pair details also include `numeric_agreement` (0–1, or `null`
when neither output has numbers), alongside token and sequence scores.

### Evaluate saved repository outputs

Run the existing saved Mistral, openai-6, fable-5-1 and MinerU outputs through
the evaluator without making new OCR requests:

```bash
conda run --no-capture-output -n myenv3.13 python scripts/cross_model_eval.py
```

Results are saved in `output/cross_model/`: `report.md`, `results.json`,
`ranking.csv`, `page_differences.json`, `text_volume.json`, and
`normalized_pages.json`. The JSON includes overall and per-document rankings,
pairwise scores, and provenance. Differences include text passages and numeric
strings, preserving leading zeroes in the numeric comparison.
Normalized page exports include `input_format: "plain"` so they can be supplied
back through `--input` without parsing literal markup again. Repository loading
requires exactly one matching output file/run per method and document; missing
or ambiguous matches raise an error instead of choosing the first result.
For multiple runs, supply exact page paths to `score_page()` or prepare a custom
`--input` JSON from the chosen run. Selected raw OCR artifact hashes are saved
in the provenance metadata.

The ranking measures **consensus, not accuracy**: each pair compares whole
text regions located by ontology labels. Token,
label-weighted, sequence and numeric agreement remain separate diagnostics.
Each method's score averages its agreement with the other methods on each page,
then averages pages within each document and gives each document equal weight.
Formatting is removed before comparison; layout and structure are not scored.
Shared OCR errors can inflate consensus. The saved openai-6 and fable-5-1
artifacts describe Tesseract with visual review, so those folder names do not
verify inference by the named models.

Options include `--weighting page`,
`--documents csa board trade`, `--models mistral openai-6 mineru`, and
`--output output/custom_evaluation`.
Use `--overlap-weight 0.7` only to explicitly opt into a 70/30 overlap/sequence
blend; the default 1.0 compares regions. Region alignment allows passages to move;
ordered bigrams retain sensitivity to local word and number order.

To evaluate other saved text, provide `--input path/to/pages.json` using this
schema (at least three methods are required):

```json
{
  "document_1": {
    "method_a": {"pages": ["Page one text", "Page two text"]},
    "method_b": {"pages": ["Page one text", "Page two text"]},
    "method_c": {"pages": ["Page one text", "Page two text"]}
  }
}
```

Every document must contain the same methods with matching page counts. Align
the same physical pages in the same order before evaluation; matching counts
alone cannot verify alignment. All-empty pages are excluded. A pair of empty
outputs receives zero agreement when another method contains text on that page.

Custom JSON records may specify `"input_format": "plain"`, `"markdown"`
(default), or `"html"`. Use `plain` for text already extracted from formatting,
including previously normalized text, so literal markup is not parsed again.

Run the evaluation tests with:

```bash
conda run --no-capture-output -n myenv3.13 python -m unittest discover -s tests -p 'test_cross_model_eval.py'
```

## Code files

- `scripts/ocr_common.py`: discovery, image/PDF decoding, caching and output.
- `scripts/cross_model_eval.py`: configurable text consensus evaluation CLI.
- `scripts/run_mineru.py`: local model initialization and extraction.
- `scripts/run_mistral_ocr.py`: remote API requests and response conversion.
- `scripts/setup_mineru.sh`: optional local model installation.
- `requirements-mineru.txt`, `requirements-mistral.txt`: backend dependencies.

References: [Mistral OCR](https://docs.mistral.ai/studio/document-processing/basic_ocr),
[MinerU model](https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B).

## Ontology region evaluation (algorithm 6.1)

With an ontology, the default primary score compares **whole text regions** rather
than extracted field values. Repository scripts use `document_ontology.json`;
the standalone library requires `ontology_path` / `--ontology` and otherwise
uses ordinary token overlap.

1. Match ontology concept names, children, fields and declared aliases. CamelCase
   and spaced forms share canonical anchor tags. Suppress common function-word
   labels such as `To` and labels shorter than three characters.
2. Select a window of 16 tokens before and after each anchor. Keep boundaries
   relative to that anchor; there are no fixed-size document chunks. Divide each
   token's weight equally among its covering windows, and each bigram's weight
   among windows containing both tokens. Overlap never duplicates selected mass.
3. Align windows one-to-one by maximum weighted agreement within the same central
   canonical anchor tag. For anchors with ambiguous owners, require identical
   window tokens, a shared unambiguous neighbouring anchor, or at least two shared
   context words with word Dice of at least 0.5. Context words exclude anchors,
   numbers, common function words and words shorter than three characters.
   Repeated regions cannot reuse a peer region. A generic shared label alone
   cannot connect unrelated passages.
4. Score each matched region using `0.5 * token Dice + 0.5 * ordered bigram Dice`,
   applying the overlap weights to repeated occurrences. Weight matches by their combined token mass;
   divide matched mass by all selected token mass. Unmatched regions receive zero.

Every word and numeric token within a region contributes. Signs, separators,
percentages and leading zeroes remain meaningful. Dates and numbers are not typed
or normalized as exact values. Labels are canonicalized to one anchor token;
other punctuation is ignored. Matching labels alone can still agree even when no
value is present: this measures selected-text consensus, not field completeness.

`ontology_region_agreement` is the primary metric. `ontology_anchor_agreement`
compares anchor occurrences separately. Per-output `localization` reports anchor
counts, ambiguous anchors, region counts, selected/total tokens and the selected
fraction. The pair diagnostics include region passages, source character/line
boundaries, anchors, overlap weights, match details and unmatched regions. Rejected
context candidates are counted, with up to 20 examples per pair. Ambiguous owners
do not incur an automatic penalty; they require evidence for correspondence.
Exact field extraction is not called. Algorithm 6.0 scores must be regenerated.

```bash
conda run --no-capture-output -n myenv3.13 python scripts/score_page.py a.md b.md --ontology-mode regions --region-context-tokens 16
conda run --no-capture-output -n myenv3.13 python scripts/cross_model_eval.py --input aligned.json --ontology document_ontology.json --region-context-tokens 16
```

Python APIs accept `region_context_tokens=16` (integer 1–128). Larger contexts
include more neighbouring text; smaller contexts can miss a value. Pages with no
located regions across all peers are excluded and reported; entirely unscorable
inputs raise an error. A peer without regions receives zero when others have
regions. No token-score fallback inflates region agreement. Inspect coverage:
misspelled/undeclared labels may prevent localization, and dense ontologies can
select much of a page. Anchor consensus cannot establish ontology quality without
reviewed reference annotations.

These are **text regions**. Spatial bounding boxes, image coordinates and exact
field correctness are not evaluated. Regenerate saved metrics before ranking;
old field-value or label scores are rejected by the region ranking script.

## Optional exact field-value evaluation

Select `ontology_mode="values"` / `--ontology-mode values` to retain the previous
exact field-value algorithm. This mode uses the following extraction policy.

Extraction uses explicit `Label: value` / `Label = value` pairs, typed values after
a label (`Amount 100`), adjacent label/value lines, and Markdown/HTML/tab-separated
tables. CamelCase, spaced names and declared aliases map to one canonical field.
Shared field names retain their owning concept when a section heading supplies
context; ambiguous owners are reported without match credit. Unlabelled values,
misspelled labels and undeclared synonyms are not inferred.

Each extracted record contains the canonical field, record key, occurrence,
raw/normalized value, value type, validity, source line and source text. Stable
unlabelled first-column table keys identify records; otherwise repeated fields
use occurrence order. Swapping `Amount 100 Account 200` to `Amount 200 Account 100`
scores 0%, even though the token counts match. `Amount 100` versus `Amount 900`
also scores 0%. Amount/currency associations are preserved within row/scope and
occurrence. Reordering rows without stable keys can reduce agreement.

`ontology_field_value_agreement = 2 * matching_valid_records / total_records`

All extracted occurrences count equally. Missing and extra fields lower the
score; invalid values and extraction issues add to the denominator and receive
no match credit. Two outputs with no fields have `null` field agreement and
receive zero in a page where another output has fields. Batch pages with no field
evidence are excluded; an entirely unscorable batch or single page raises an error.
No token-overlap fallback inflates field-value agreement.

Types are inferred from common field names and can be overridden:

```json
{
  "date_order": "dmy",
  "concepts": {
    "Invoice": {
      "value_type": "identifier",
      "fields": ["Total", "AccountNumber", "SettlementDate", "Currency"],
      "field_types": {"Total": "amount", "AccountNumber": "identifier"}
    },
    "Company": {"aliases": ["Business Name"], "value_type": "text"}
  }
}
```

Supported types are `text`, `identifier`, `number`, `amount`, `date`, and `currency`.
Identifiers preserve leading zeroes and case; standard IBAN/BBAN/BIC/ISIN/LEI/UETR
labels additionally normalize spaces and letter case. Numbers use exact Decimal
comparison, dot decimals and correctly grouped thousands commas. Signs,
accounting negatives, percentages and currencies remain meaningful. Dates support
ISO dates, English month names and unambiguous day/month forms; ambiguous dates
require `date_order: "dmy"` or `"mdy"` globally or on the owning concept. Text
comparison normalizes whitespace, Unicode and case. This does not validate bank
identifiers, currency membership or correctness against the source document.

```bash
conda run --no-capture-output -n myenv3.13 python scripts/score_page.py a.md b.md --ontology-mode values
conda run --no-capture-output -n myenv3.13 python scripts/cross_model_eval.py --input aligned.json --ontology document_ontology.json --ontology-mode values
# Explicit legacy label-weighted scoring:
conda run --no-capture-output -n myenv3.13 python scripts/score_page.py a.md b.md --ontology-mode mentions --entity-weight 3
```

The Python APIs accept `ontology_mode="regions"` (default), `"values"` or `"mentions"`.
`entity_weight` affects only mentions mode and must be finite and at least 1;
weight 1 reproduces unweighted overlap in that mode. Legacy token and label scores
remain available in diagnostics. Ontology SHA-256 and scoring settings are saved.
Field diagnostics include `per_field_agreement`, matched counts, unmatched records,
invalid-value reasons and ambiguous-label/column-count issues.

Scores from these modes must not be mixed or interpreted as region scores. Consensus still requires independent reviewed
reference values before it can be interpreted as accuracy. Extraction coverage
is limited to supported labels and structures; inspect issues and extracted values
when applying the full ontology to a new document family.
