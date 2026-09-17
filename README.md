# OCR evaluation

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
The score measures consensus, not accuracy. Version 4 uses ontology-weighted
overlap by default (`overlap_weight=1.0`), so moving intact phrases does not lower the
primary score. Omissions, extra repetitions and numeric changes still count.
For diagnostics, `score_page_details(paths)` returns `score`, `model_scores`
(in input order), and `pairs` with separate token, sequence and numeric agreement.
Pair similarities are 0–1; the page and method scores are 0–100. Sequence and
numeric diagnostics have no additional weight in the default primary score.
Explicitly setting `overlap_weight=0.5` opts into a 50/50 weighted-overlap/sequence blend.

Version 3.1 preserves accounting parentheses as written: `(100.00)` differs
from `100.00` without assuming every parenthesized number is negative. Numeric
agreement now counts occurrences, so missing a repeated amount lowers it;
`unmatched_left_numeric_counts` and `unmatched_right_numeric_counts` identify
the missing or extra occurrences. Inline HTML emphasis does not split words,
while block elements and table cells remain separated.

The default `score_page()` and `score_models()` use a phrase-and-counter path, avoiding
sequence alignment and full difference generation. Request
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

The ranking measures **consensus, not accuracy**: each pair uses ontology-weighted
overlap, with ontology concept/field mentions weighted 3× (see the formula below). Sequence and numeric agreement are reported
separately. Identical token counts can score 100 even if word associations differ.
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
blend; the default 1.0 ignores global order while matching contiguous ontology phrases.

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

## Ontology weighting (algorithm 4.0)

The default evaluation now gives mentions of concepts and fields in
`document_ontology.json` **3× weight**, versus 1× for ordinary tokens. Page scoring,
cross-model evaluation, and the five/six-model comparisons use the same metric.
Regenerate saved metrics before running `rank_ocr.py`; old results are not migrated.

Matching uses case-insensitive whole-token phrases, including CamelCase and spaced
names (`SettlementDate` / `Settlement Date`). Longest nonoverlapping matches prevent
nested concepts and duplicate ontology entries from multiplying the weight.
Definitions, evidence, and metadata are not interpreted as instructions or terms.
This is lexical mention matching, not named-entity recognition: unlabeled values,
synonyms, and misspelled labels are not inferred. Reordering intact phrases is free;
breaking a phrase can change its classification. Entity values such as account
numbers retain ordinary weight unless they themselves match the vocabulary.

Let T be total token occurrences across a pair, S their shared count, E the total
number of tokens covered by matched phrases, and M the shared phrase count weighted
by phrase length. Shared counts use the minimum occurrence count on either side.
The weighted overlap is `2 * (S + (w - 1) * M) / (T + (w - 1) * E)`.
For `IBAN hello` versus `typo hello`, the weighted score is 33.33%; changing the
ordinary word instead (`IBAN typo`) gives 75%. Both unweighted scores are 50%.

```bash
conda run --no-capture-output -n myenv3.13 python scripts/score_page.py a.md b.md --entity-weight 3
conda run --no-capture-output -n myenv3.13 python scripts/cross_model_eval.py --input aligned.json --ontology document_ontology.json --entity-weight 3
```

Python APIs accept `ontology_path=...` and `entity_weight=...`. Weight must be finite
and at least 1; **weight 1 reproduces unweighted overlap**. The ontology must exist
and be valid; it is never silently skipped. Detailed results record its SHA-256,
matching policy, and weight. Diagnostics include `ontology_weighted_overlap`,
`ontology_entity_agreement` (null when neither text has mentions), matched/unmatched
phrase counts, and the original `token_overlap`. Scores remain consensus rather
than accuracy against ground truth.
