# OCR evaluation

## Unified comparison of all saved workflows

Use this command for the current cross-model comparison; the experiment-specific
reports below are historical snapshots with different peers and scoring versions.

```bash
conda run --no-capture-output -n myenv3.13 python scripts/evaluate_all_saved.py
```

The [unified report](output/unified_corroboration_2families_v2/report.md) and
[comparison CSV](output/unified_corroboration_2families_v2/comparison.csv) cover 12 saved workflows,
including both Unlimited-OCR variants, Claude's partial run and saved Tesseract
hybrid, and correctly labelled historical Tesseract outputs. This is offline
evaluation of existing artifacts, with no OCR/API requests.

- One unchanged 27-field reference pilot, equal document weighting, and physical
  page routing for every aligned output. Benchmark 1.1 rejects duplicate answers
  for a record/field instead of choosing the answer closest to the reference.
- Full 48-page results retain failed pages as empty and truncated text as saved.
  A separate identical 45-page completed subset is used for all aligned workflows.
  Coverage and complete-document success are shown alongside selected-field scores.
- Agreement is a secondary diagnostic: equal token/ordered-bigram Dice, a fixed
  panel, and equal engine-family votes excluding relatives of the candidate.
  Additional candidates do not become voters. Tokenization is independent per
  output; the evaluator avoids global sequence alignment and candidate-only pairs.
- Passage corroboration is a separate diagnostic using monotonic token alignment
  within physical pages. [Passage CSV](output/unified_corroboration_2families_v2/passage_corroboration.csv)
  lists family support counts, source text, context, offsets, uncertainty reasons and separate flags;
  [uncertain passages](output/unified_corroboration_2families_v2/uncertain_passages.csv) filters
  below-threshold text, while [flagged passages](output/unified_corroboration_2families_v2/flagged_passages.csv)
  includes disagreements and alignment concerns even on corroborated text. The matching JSON includes every panel member's alternate
  reading and normalized token span. Passage rows are per source workflow, not
  deduplicated ground-truth regions.
- Insavlo lacks physical page boundaries and remains available only in the
  document-agreement and unbound-extraction diagnostics, not page-bound accuracy.
- Selected artifact, source, evaluator and policy hashes are locked in
  `run_manifest.json`. Identical reruns are allowed; changed inputs or code require
  a different `--output` directory. Existing historical outputs are preserved.

This remains an exploratory sample of seven pages from three documents, with
post-hoc extraction rules. More models do not make it a held-out benchmark.
Independent reference review and additional documents cannot be supplied by
rescoring the existing outputs. See the report's provenance caveats and
[field audit](output/unified_corroboration_2families_v2/field_audit.csv).

Passage corroboration defaults to **two independent engine families**, with no
proportional floor (`min_families=2`, `min_fraction=0`). Every configured
panel member of a family must have completed the page and support the exact
normalized passage for that family to vote. Family disagreements are flagged;
failed/truncated members abstain and the denominator does not shrink. A source
counts only if it belongs to the fixed panel. `other_family_support_count`
separately excludes the source's family ancestry. Non-panel candidates never vote.

The threshold is configurable with `--corroboration-min-families`,
`--corroboration-min-fraction` and `--passage-tokens` (default 40, maximum 128).
The fraction default of zero disables the proportional requirement; a nonzero
fraction raises the threshold to the larger of the count and the rounded-up
fraction of configured families. The older three-family/50% policy remains
available by explicitly passing `--corroboration-min-families 3
--corroboration-min-fraction 0.5` with a new output directory.
Passages follow decoded source lines, split at the token limit; peer line wrapping
does not need to match. Matching ignores case, formatting and ordinary punctuation,
preserves signed numeric tokens, and does not use fuzzy matches or peer repairs.
Repeated matches need unique matching context, up to eight tokens on either side.
Offsets are zero-based, end-exclusive positions in normalized page tokens, not
image coordinates or raw character offsets.

Meeting the support threshold marks a passage **corroborated**, even when other
families disagree. Only below-threshold support makes a passage **uncertain**.
Conflicting readings, ambiguous alignment and incomplete source output remain in
the separate `flags` list; they do not veto corroboration. A three-versus-three split
can corroborate both readings, with both retaining conflict flags. Empty outputs are recorded separately in the JSON as
uncertain, never verified blank. The threshold is an uncalibrated review heuristic;
family separation does not prove independent errors, and jointly omitted text
cannot be found from these outputs. Passage counts depend on source formatting and
must not be used as model rankings. Existing agreement/accuracy scores are unchanged.
Use a new output directory when changing code or policy; older reports are preserved.

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

To reproduce the exploratory evaluation of the six saved outputs:

```bash
conda run --no-capture-output -n myenv3.13 python scripts/evaluate_six_critical_fields.py
```

The [pilot report](output/six_critical_field_evaluation/report.md) covers 27
image-checked fields across seven pages, with a complete field audit and engine-family
consensus sensitivity. It measures OCR plus shared extraction on these documents;
it is not a held-out or independently human-adjudicated benchmark. The script
checks source and saved-artifact hashes and retains incorrect OCR values.

## Shared invoice extractor (version 3)

```bash
conda run --no-capture-output -n myenv3.13 python scripts/reevaluate_shared_extractor.py
```

The shared pilot extractor supports invoice headers emitted as a label column
followed by a value column: unique `DOC NO.`, `DATE`, and optional `GST NO.`
labels pair by row order with an equal number of colon-prefixed value rows.
Ambiguous or incomplete columns remain unextracted. Existing inline and wrapped
single label/value rows retain their behavior. Values and decoded spans are
preserved verbatim; no reference values or digit/date corrections are used.
The rule remains confined to the first invoice's header.

The 2026-10-12 [version 3 evaluation](output/extractor_v3_evaluation/report.md)
reextracts every model in all six saved pilot comparisons from verified OCR
artifacts, with unchanged references, scoring and document subsets. It preserves
the previous outputs and makes no OCR/API requests. Claude is evaluated on its
completed board/trade subset (21 fields); full-corpus methods use 27 fields.
Both Unlimited-OCR variants and the historical six-method outputs are covered.

Rotated Unlimited-OCR recovers the invoice number and date, improving
**23/27 → 25/27** and **86.54 → 94.87/100**. Its remaining misses are the two
board schedule fields. All other saved models retain their prior scores under
the same rules. Whole-text agreement remains **90.34/100** for the rotated run,
since this changes extraction rather than OCR. The report and scenario-specific
field audits retain before/after predictions. This is a post-hoc convenience
sample, not a held-out accuracy benchmark.

## Standalone Python library

See [standalone_ocr_eval/README.md](standalone_ocr_eval/README.md) for installation, API, CLI, input schema, and optional ontology instructions.
Version 0.5 includes portable `evaluate_saved` and `ocr-eval evaluate-saved` entry
points for full-text agreement without ontology and two-family passage corroboration.
It requires no OCR backend or repository data; the input bundle supplies saved
text, engine families and the fixed voting panel.

## Verified OpenAI VLM OCR (GPT-6 Astra)

The historical `output/openai-6/` artifacts identify Tesseract with selective
review. A genuine OpenAI VLM run is saved separately and labelled `gpt-6-astra`.

Set `OPENAI_API_KEY` locally, then run:

```bash
conda run --no-capture-output -n myenv3.13 python -m pip install --no-user -r requirements-astra.txt
conda run --no-capture-output -n myenv3.13 python scripts/run_astra_ocr.py --workers 3 --output output/openai-vlm-gpt-6-astra
conda run --no-capture-output -n myenv3.13 python scripts/evaluate_openai_vlm.py
```

The runner submits rendered page images to the official OpenAI Responses API;
this uploads the documents and incurs API charges. It pins the endpoint to
`https://api.openai.com/v1`. Defaults: all three current PDFs (48 pages), 144 DPI,
`detail=high`, low reasoning effort, structured JSON and 16,000 maximum output
tokens per page. `--detail original` is available for a separate experiment.
`--workers` controls concurrent documents (1–3); PDFium rendering calls are
serialized for thread safety. `--input`, `--output`, `--dpi`, `--dry-run` and
`--overwrite` are also supported.

Each completed page retains the raw API response, returned model ID, response
ID, request timing, image hash/dimensions, ordered blocks and Markdown. Source
bytes, prompt, schema and inference settings determine the cache directory.
Incomplete or unexpected-model responses are saved as `incomplete-NNNN.json`
and cannot be reused as completed checkpoints. The manifest records completed
documents and failures; rerunning resumes saved pages. Do not run concurrent
commands against the same output directory.

The evaluator verifies complete coverage and artifact hashes, adds the actual
VLM alongside six saved workflows, excluding the historical `openai-6` output, and
generates **whole-document token agreement without ontology**, with equal
document weighting and engine-family sensitivity. It also reuses the frozen
27-field pilot with the same references and versioned shared extraction rules for every model. The remaining
`fable-5-1` baseline is labelled as a Tesseract-based artifact.

Results: [comparison report](output/openai_vlm_experiment/report.md),
[ranking](output/openai_vlm_experiment/ranking.csv),
[field audit](output/openai_vlm_experiment/field_audit.csv).

The 2026-10-11 API run completed all 48 pages, each returning `gpt-6-astra`
with completed status. After excluding `openai-6`, without ontology it scored
**91.54/100**, ranking third among seven outputs after the PaddleOCR rerun
(CSA 98.42, board 86.54, trade 89.67). It passed
**27/27** frozen pilot fields for a 100/100 pilot score. This selected-field
result does not establish full-corpus accuracy. Run wall time was 345.16 seconds
with three document workers; usage was 128,672 input and 33,564 output tokens.

API references: [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra),
[image inputs](https://developers.openai.com/api/docs/guides/images-vision).

## Claude OCR through a compatible gateway

`scripts/run_claude_ocr.py` runs `claude-fable-5-1` at the user-selected
`https://one.liangzhili.com/v1` endpoint. Model identity is gateway-reported;
the upstream provider is not independently verified. This is separate from the
historical Tesseract-based `fable-5-1` output.

Set `OPENAI_API_KEY` locally. The existing OpenAI SDK dependency is reused:

```bash
conda run --no-capture-output -n myenv3.13 python scripts/run_claude_ocr.py --json-mode json_object --workers 3
conda run --no-capture-output -n myenv3.13 python scripts/evaluate_claude_vlm.py
```

The runner uses the same three PDFs, 144-DPI images and transcription system
prompt as the OpenAI experiment. The gateway did not enforce strict schema mode
in a synthetic test, so this run uses JSON-object mode with the schema in the
user message. Reasoning uses provider defaults. Saved native responses must
match the requested model, finish normally and satisfy the schema before being
accepted. No text repairs are applied. HTTP 400 page failures are retained and
the remaining pages continue; reruns reuse both successes and recorded failures.

The evaluator rejects incomplete runs by default. If pages are blocked, use
`--complete-documents-only` to compare only fully completed documents, removing
the same documents from every peer. This subset is explicitly labelled and must
not be compared directly with earlier full-corpus scores. Both modes exclude
`openai-6` and use token overlap without ontology and the same versioned shared
field extractor for every method.
Outputs are saved under `output/claude-vlm-fable-5-1/` and the
[comparison report](output/claude_vlm_experiment/report.md).

The 2026-10-11 gateway run returned valid OCR for 46/48 pages; CSA pages 2 and 3
were blocked by the gateway's content filter. The shared **34-page board/trade
subset** therefore excludes CSA from all seven methods. After removing the
historical Tesseract Fable output and refreshing PaddleOCR, Claude scored **88.21/100** in token agreement
(rank 5/7) and passed **21/21** selected fields. Recalculate exclusions from saved
inputs with `scripts/evaluate_claude_vlm.py --rescore-existing`; this preserves
the original extraction even if the shared extractor has subsequently changed.
These subset scores are not directly comparable with the earlier three-document
scores. Saved response image hashes match the OpenAI experiment for all 46
successful pages.

## PaddleOCR v4 (local CPU)

```bash
bash scripts/setup_paddleocr.sh
conda run --no-capture-output -n myenv3.13 python scripts/evaluate_six_critical_fields.py
conda run --no-capture-output -n myenv3.13 python scripts/run_paddleocr_v4.py --no-textline-orientation
conda run --no-capture-output -n myenv3.13 python scripts/evaluate_paddleocr_v4.py
```

Runs the English **PP-OCRv4 models**, explicitly selected through PaddleOCR
3.7.0 / PaddlePaddle 3.3.1 / PaddleX 3.7.2. The OCR package version and model
generation are different. Uses `PP-OCRv4_mobile_det` and
`en_PP-OCRv4_mobile_rec`, plus PP-LCNet document orientation for rotated scans.
Text-line orientation is disabled by default after it incorrectly rotated upright
board date/signing lines by 180 degrees. Use `--textline-orientation` to enable
it for a comparison run. Document unwarping is disabled. Model weights download from
official Paddle hosting; document inference stays local.

Setup installs packages in `.cache/paddleocr-packages`, loaded only by this
runner through the existing `myenv3.13` Python. It does not replace the packages
used by other OCR runners or create another environment. Installation requires
compatible binary wheels and fails rather than switching Python versions.

Defaults process all three PDFs (48 pages) at 144 DPI on CPU with 8 threads,
a 2048-pixel maximum detector side, and recognition confidence threshold 0.5.
Outputs under `output/paddleocr-v4/` include page checkpoints with native JSON
(text, confidence scores, polygons and orientation), plain `document.txt`,
`source.json`, model hashes, resolved pipeline settings, environment versions,
and a manifest selecting completed runs. Text remains in native recognition
order with one detection per line; tables are not reconstructed. Polygon
coordinates refer to the image after automatic document rotation.

The runner supports `--dry-run`, `--input`, `--output`, `--limit`, `--overwrite`,
`--dpi`, `--det-limit`, `--cpu-threads`, `--textline-orientation`,
`--no-textline-orientation`, `--smoke-test`, and `--setup-only`.
Settings and source changes create new run directories; interrupted runs resume
from page checkpoints. Avoid concurrent writes to the same output root. A run's
manifest contains only that invocation's selected files; rerun the default full
corpus command before evaluating after a partial-file experiment.

The evaluator requires all 48 pages, verifies source/settings/artifact hashes,
and compares the new output with the six saved methods. It keeps the existing
27-field pilot references and comparison policies unchanged. Shared extractor
version 2 tolerates spacing, sentence-ending account punctuation, joined
currency/amount tokens and schedule anchoring, preserving raw values and spans.
All methods are re-extracted with these same rules. The evaluator separately
recomputes whole-document consensus and engine-family sensitivity. Pilot scores
measure OCR plus extraction on a convenience sample, not held-out OCR accuracy;
consensus scores measure peer agreement. See
[experiment report](output/paddleocr_v4_experiment/report.md),
[field audit](output/paddleocr_v4_experiment/field_audit.csv), and
[page timings](output/paddleocr_v4_experiment/page_timing.csv).

The 2026-10-11 rerun with text-line orientation disabled completed all 48 pages
with no failures in 456.76 seconds of page inference (9.52 seconds/page,
excluding setup and rendering). PP-OCRv4 passed **26/27** pilot fields:
board 12/13, CSA 6/6, trade 8/8. Its equal-document pilot score is
**97.44/100**, tied for second with Mistral in the seven-output comparison.
Whole-document ontology-region agreement is **67.55/100** (7th); whole-text
token agreement is **86.87/100** (6th). These are different metrics, and peer
agreement is not reference accuracy.

The prior score was 70.73 (18/27). Applying extractor version 2 to that same
OCR text raises it to 94.87 (25/27); the new inference raises it to 97.44
(26/27). All six historical methods were re-extracted with the same rules and
retain their previous critical-field scores. The remaining PaddleOCR failure
is the schedule signing rule, read as `CATEGORY A &B` rather than
`CATEGORY (A & B)`. The raw value is retained and this text field still fails
the existing comparison. Typed month dates accept optional space after the
comma; date validity and literal invoice-date comparison remain enforced.
These are post-hoc exploratory scores, not held-out accuracy.

The [change comparison](output/paddleocr_v4_experiment/change_comparison.json)
separates extraction improvements from inference changes. Previous results
are preserved under `output/extractor_v1_orientation_enabled_archive/`, and
the original OCR checkpoints remain in their settings-specific directories.

Model/API reference: [PaddleOCR general OCR pipeline](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html).

## Unlimited-OCR (local Apple Silicon, BF16)

```bash
# Setup is optional when the existing DeepSeek MLX-VLM 0.7.0 packages are present.
bash scripts/setup_unlimited.sh
conda run --no-capture-output -n myenv3.13 python scripts/run_unlimited_ocr.py
conda run --no-capture-output -n myenv3.13 python scripts/evaluate_unlimited_ocr.py
```

Runs [Baidu Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) through
the unquantized [MLX BF16 conversion](https://huggingface.co/mlx-community/Unlimited-OCR-bf16),
pinned to revision `6d9f675e3fa73dd49cd03f630868b1941c72803f`. This is an
Apple Silicon runtime experiment, not the official CUDA implementation.
Requires Metal GPU access and approximately 6.22 GiB of downloaded weights.
All Python runs use `myenv3.13`; no additional environment is created.

Defaults: all three PDFs (48 pages), 144 DPI, single-page `document parsing.`
prompt, gundam mode (1024 global view and 640 local crops), greedy decoding,
32768 output tokens/page and Baidu's sliding no-repeat n-gram guard (35/128).
It does not test the model's one-shot multi-page capability. Supports `--input`,
`--output`, `--limit`, `--dpi`, `--max-tokens`, `--dry-run`, `--setup-only`,
and `--overwrite`. Source/settings hashes select resumable directories;
incomplete generations are retained and rejected for scoring.

Outputs under `output/unlimited-ocr-bf16/` retain raw layout-marked model text,
cleaned Markdown, image hashes/dimensions, token counts, completion status,
timings, model file hashes and runtime versions. Inference stays local.
The evaluator verifies all 48 page artifacts and re-rendered image hashes,
then compares with saved Mistral, MinerU, Insavlo, DeepSeek, PaddleOCR v4
and genuine GPT-6 Astra outputs. Historical Tesseract-labelled `openai-6`
and `fable-5-1` are excluded. DeepSeek and Unlimited-OCR share a conservative
family group for the separate family-sensitivity calculation.

The [experiment report](output/unlimited_ocr_experiment/report.md),
[ranking](output/unlimited_ocr_experiment/ranking.csv),
[field audit](output/unlimited_ocr_experiment/field_audit.csv), and
[page timings](output/unlimited_ocr_experiment/page_timing.csv) distinguish
whole-document token agreement without ontology from the unchanged 27-field
OCR-plus-extraction pilot. Peer agreement is not accuracy; the pilot is a
convenience sample, not a held-out accuracy benchmark.

The 2026-10-12 BF16 run completed all 48 pages with no runtime failures in
303.01 seconds (296.92 seconds of page inference; 44,114 generated tokens).
Whole-document token agreement without ontology was **87.05/100**, ranking
**5/7** in this comparison. It passed **22/27** frozen pilot fields:
board 11/13, CSA 6/6, trade 5/8, with an equal-document pilot score of
**82.37/100**. Board page 4 was classified entirely as an image; the rotated
trade invoice on page 9 produced erroneous text. Their five selected fields
are missing. Generation stop status does not establish transcription quality.

To reproduce the separate rotation-correction experiment, reuse the verified
PP-LCNet document-angle predictions from the existing PaddleOCR checkpoints:

```bash
conda run --no-capture-output -n myenv3.13 python scripts/run_unlimited_ocr.py --orientation-manifest output/paddleocr-v4/manifest.json
conda run --no-capture-output -n myenv3.13 python scripts/evaluate_unlimited_ocr.py --ocr-output output/unlimited-ocr-bf16-rotated --output output/unlimited_ocr_rotation_experiment --compare-ocr-output output/unlimited-ocr-bf16 --compare-experiment output/unlimited_ocr_experiment
```

Trade pages 9 and 10 are rotated losslessly 90 degrees clockwise; the other
46 images are unchanged. This replays only verified classifier angles, not
PaddleOCR transcription, with original source hashes and matching render DPI.
It does not deskew pages. Board page 4 is already upright and is left unchanged.
The variant saves separately under `output/unlimited-ocr-bf16-rotated/` and
preserves the original run. The [rotation report](output/unlimited_ocr_rotation_experiment/report.md),
[recovery audit](output/unlimited_ocr_rotation_experiment/recovery_audit.csv),
and [page changes](output/unlimited_ocr_rotation_experiment/page_changes.csv)
compare identical peers, inference settings, references and extraction rules.
Variant inference timings exclude the cached orientation classifier run.

The 2026-10-12 rotation rerun completed all 48 pages without runtime failures
in 286.61 seconds. Whole-text agreement rose **87.05 → 90.34/100** (still 5/7).
The unchanged field extractor improved **22/27 → 23/27**, with an equal-document
score of **82.37 → 86.54/100**. All three invoice reference values are now
present in raw OCR on trade page 9. The account is extracted correctly; number
and date remain unextracted because their labels are emitted separately from
their values. The two board schedule fields remain absent. All 46 unrotated
pages reproduced identical raw text. This post-hoc ablation preserves the
original pilot and does not grant extraction credit for literal presence alone.

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

## API LLM schema selection and critical-field identification

The `identify-critical-fields` command uses an OpenAI-compatible Chat Completions
endpoint in two stages: the model selects relevant ontology schemas, then identifies
critical fields and their verbatim evidence. It accepts one OCR Markdown/text or JSON
document, without a reference transcription or source image. Existing consensus and
reference-benchmark commands retain their behavior.

Compile the full ontology once (streams the repository's large JSON export):

```bash
conda run --no-capture-output -n myenv3.13 python scripts/benchmark_ocr.py compile-schemas \
  --ontology /Users/yimingqian/Documents/GitHub/Finance_OCR_Benchmark/output/phase5_refiltered/ontology.json \
  --output output/llm_critical_fields/iso20022_schemas.json
```

The compiler also accepts `document_ontology.json`, or an explicit catalog with
`schema_version: 1` and a `schemas` list. Each schema has a unique `id`, `name`,
`definition`, and `fields` list; each field has an `id`, `name`, and optional
`definition`. The full graph supplies direct business/message owner fields, including
unmapped ones, with source IDs, definitions, registration status, and selected type
and cardinality attributes. It does not infer inherited fields or translate every
ontology edge into a constraint. The compact export lacks field definitions.

Set `OPENAI_API_KEY` in your environment. Supply the model explicitly with `--model`
or `OPENAI_MODEL`. Set `OPENAI_BASE_URL` or `--base-url` for another compatible server;
the default is `https://api.openai.com/v1`. The server must support `/chat/completions`
and `response_format: {"type":"json_object"}`. No new package is required.

```bash
conda run --no-capture-output -n myenv3.13 python scripts/benchmark_ocr.py identify-critical-fields \
  output/six_critical_field_evaluation/decoded/mistral/board.txt \
  --schemas output/llm_critical_fields/iso20022_schemas.json \
  --model "$OPENAI_MODEL" \
  --output output/llm_critical_fields/mistral_board.json
```

Use `--policy policy.txt` to specify financial criticality criteria. The disclosed
default considers party identity, routing, monetary obligations, operative dates,
authorization, and transaction interpretation. These are model-proposed decisions,
not domain-expert approval. ISO field membership and minOccurs do not establish
criticality or document requiredness.

Processing and output contract:

- All text is processed in overlapping character chunks (`--chunk-chars 12000`,
  `--overlap 1000`). JSON values retain RFC 6901 pointers; nearby scalar properties
  supply limited context. Run each OCR document separately, not a JSON container
  combining engines or reference answers. For structured JSON, a parent object's
  scalar context is limited to 4,000 characters; full sibling/table relationships
  may require preprocessing into text blocks.
- Local lexical retrieval shortlists `--top-k 20` schemas per block; the LLM makes
  the final selection, including abstention and uncovered topics. Retrieval is a
  bounded candidate stage, not exhaustive coverage or embedding search. Increase
  top-k to assess retrieval sensitivity. Selected fields are batched in groups of
  `--field-batch 40`, with no silent field-list truncation.
- `routes` records schema candidates and selections. `decisions` records critical
  and noncritical proposals, rationales, and block-local found/unlocated/ambiguous
  states. Omitted candidates are unassessed, not negative labels.
- `fields` contains only occurrences whose exact quoted evidence is found in the
  source block and whose value occurs once in that quote. Offsets are computed by
  code, zero-based and end-exclusive in Unicode characters. Markdown offsets refer
  to the original UTF-8-decoded text, including CRLF; JSON offsets refer to the
  decoded string. For JSON numbers/booleans, the pointer identifies the value and
  offsets refer to its JSON scalar rendering, not raw-file byte positions.
- Rejected evidence is preserved in `issues`. Raw model proposals remain in
  `decisions` and the API trace, and must not be treated as verified occurrences.
  `source_verified` proves literal membership only, not correct semantic binding
  or original-image fidelity. Overlapping identical field spans are deduplicated;
  different schema mappings and conflicting criticality judgments remain visible.
- Output includes input/catalog/ontology/policy fingerprints, prompt version,
  request hashes, model responses, usage, and model identity. API keys are not
  stored. These audit outputs contain source excerpts and extracted document data.
- HTTP rate-limit/server failures retry twice; refusals, incomplete responses,
  invalid IDs and malformed decision structures fail the run rather than silently
  switching to regex. A failed run writes no result; resume/checkpointing is not
  implemented. Model responses can vary between runs.

This command implements reference-free **identification**, not a validated OCR
quality estimator. It emits no accuracy score and does not automatically change
benchmark denominators or promote model-selected fields into reviewed profiles.
An unlocated field in one chunk is not proof that the source document omitted it.

API contract: https://developers.openai.com/api/reference/resources/chat
