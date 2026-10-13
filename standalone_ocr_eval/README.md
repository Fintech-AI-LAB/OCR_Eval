# OCR text evaluation library

Version 0.5.1 packages full-text agreement and passage corroboration with a default
threshold of **two independent engine families**. It runs offline, without ontology,
OCR inference, or third-party runtime dependencies. The
[critical-field benchmark](BENCHMARK.md) supports document
profiles, reviewed reference annotations, record-aware field metrics, fixed-region
OCR metrics, schema diagnostics, controlled variant comparisons and engine-family
consensus sensitivity. Use `ocr-eval benchmark` for reference-based accuracy.
The existing page and batch consensus APIs below keep their existing behavior.

## Replaceable LLM API for ontology-based extraction

API calls and provider response parsing are isolated in
[`ocr_eval/llm_api.py`](ocr_eval/llm_api.py), in `call_llm_api()`.
The default implementation calls an OpenAI-compatible Chat Completions endpoint,
retries transient errors, rejects refused/incomplete responses, and parses a
completed response into a JSON object. The CLI uses it automatically:

```bash
conda run --no-capture-output -n myenv3.13 python -m ocr_eval compile-schemas --ontology ontology.json --output schemas.json
conda run --no-capture-output -n myenv3.13 python -m ocr_eval identify-critical-fields ocr.txt --schemas schemas.json --model "$OPENAI_MODEL" --output extracted.json
```

Set `OPENAI_API_KEY`; `OPENAI_BASE_URL` optionally selects a compatible gateway.
Extraction proposes critical fields and verifies their literal source evidence.
It does not assign an accuracy score; reference-based evaluation needs profiles,
reviewed annotations, and predictions with document/page/record bindings.

For a different protocol, supply your own function:

```python
from pathlib import Path
import os
from ocr_eval.llm_fields import ChatClient, compile_catalog, discover
from my_provider_adapter import call_my_llm  # Your API calling/parsing function.

client = ChatClient("your-model", base_url="https://your-provider.example",
                    api_key=os.environ.get("MY_LLM_API_KEY"), request_fn=call_my_llm)
result = discover(Path("ocr.txt"), compile_catalog(Path("ontology.json")), client)
```

Your adapter receives keyword arguments `model`, `base_url`, `api_key`,
`system_prompt`, `payload`, `timeout`, `retries`, and `opener`; it may accept
`**request` or list these arguments explicitly. `payload` contains the structured
schema-selection or field-extraction task. Send it with `system_prompt` using your
provider's request format. `opener` is `None` for custom adapters.

Return a dictionary with `output` containing the parsed JSON object. Optional
metadata keys are `response_id`, `model`, `usage`, and `request_sha256`:

```python
return {"output": parsed_json_object,
        "response_id": provider_response_id,
        "model": returned_model,
        "usage": token_usage,
        "request_sha256": actual_request_hash}
```

Your adapter owns authentication, retries, response parsing, and completion/refusal
checks; raise an exception on failed, refused, incomplete, or invalid responses.
It may use its own credentials, so custom adapters do not require `OPENAI_API_KEY`.
`ChatClient.ask()` validates `output` and records the stage and selected metadata.
Schema/field ID checks, exact-source evidence validation, and chunking stay in
`llm_fields.py`. Keep credentials out of returned metadata; hash the request body
rather than headers for `request_sha256`.

## Full-text evaluation with two-family corroboration

Install this folder, then evaluate saved outputs with the portable CLI:

```bash
conda run --no-capture-output -n myenv3.13 python -m pip install --no-deps --no-build-isolation .
conda run --no-capture-output -n myenv3.13 python -m ocr_eval evaluate-saved examples/saved.json --output results
```

The input JSON has three keys: `data` (document → method → page records),
`families` (method → list of engine-family names), and `panel` (fixed voting methods).
See [the synthetic example](examples/saved.json). Each record supplies `pages`,
optional `input_format` (default `markdown`), and optional page-aligned `statuses`
(`complete`, `failed`, or `truncated`; default `complete`). Failed pages should
contain empty text, while truncated text stays as saved. Every document must
contain the same methods and aligned physical page counts. Document-only candidates
use `pages: null` plus `document_text`; they receive whole-document scores only.
Panel members must have aligned pages and a single engine family.

```python
import json
from pathlib import Path
from ocr_eval import evaluate_saved

bundle = json.loads(Path("examples/saved.json").read_text())
result = evaluate_saved(bundle["data"], bundle["families"], bundle["panel"])
```

`Score /100` is full-document agreement: 50% token Dice plus 50% ordered-bigram
Dice, averaged within other engine families and then equally across documents.
Relatives of the candidate do not vote on its score. The separate page agreement
track preserves physical page boundaries. Corroboration requires two panel families
to support an exact normalized passage; the source's family counts if it belongs
to the panel. Related variants share one family vote. These thresholds affect
passage status, not agreement scores.

The CLI writes `report.md`, `comparison.csv`, `results.json`, document/page agreement
JSON, passage corroboration JSON/CSV, `uncertain_passages.csv`, and
`flagged_passages.csv`. Original method IDs remain in exports. Passage counts depend
on line formatting and are review diagnostics rather than model rankings.

Use `--corroboration-min-families` to change the count, or
`--corroboration-min-fraction` to add a proportional floor. Its default `0`
disables that floor, keeping the requirement at exactly two families for any panel
size. For the older policy, pass `--corroboration-min-families 3
--corroboration-min-fraction 0.5`. `--passage-tokens` defaults to 40.

`ocr_eval.corroborate_saved(data, families, panel)` adds passage-level corroboration
for saved, physically aligned pages. `data` is a mapping from document to method to
`{"pages": ["page text", ...], "input_format": "plain", "statuses": ["complete", ...]}`;
statuses may be omitted for complete saved outputs. `families` maps methods to lists
of engine-family names, and `panel` fixes which methods may vote. Hybrid methods
may be evaluated but cannot vote. Document-only outputs must be excluded.

The result includes every source passage, normalized token offsets, source context,
supporting/conflicting/missing/ambiguous/unavailable families, peer readings and
uncertainty reasons and separate diagnostic `flags`. One family votes only when all its configured panel members
completed the page and support the same exact normalized passage. The source counts
only if it is in the panel; `other_family_support_count` excludes its family ancestry.
Case, formatting and ordinary punctuation are ignored; signed numeric tokens are
preserved. Monotonic alignment stays within the physical page, and repeated matches
require unique surrounding context. Source lines are split at `passage_tokens=40`.

Defaults are `min_families=2` and `min_fraction=0` (no proportional floor).
The requirement is `max(min_families, ceil(min_fraction * all_panel_families))`,
including unavailable families. Thresholds are configurable, uncalibrated heuristics.
Meeting the support threshold is sufficient for corroboration. Conflicting readings,
ambiguous alignment and incomplete source output are separate flags, not vetoes.
Only below-threshold support makes a passage uncertain; a three-versus-three split
can corroborate both readings with conflict flags. Empty page outputs are uncertain
in the separate `pages` inventory.
This is text corroboration, not verified image text or accuracy. It cannot find text
omitted by all models or establish global reading order. Existing consensus scores
and reference-based accuracy remain separate.

`ocr-text-eval` evaluates agreement among OCR transcriptions of the same pages.
It needs no OCR backend, repository data, or third-party runtime dependency.
Scores measure **consensus, not accuracy**. The ontology is optional and is not
included in the installed library; pass a compatible JSON file to evaluate its
text regions. Without one, all tokens have equal weight.

This folder is self-contained: copy it to another project or machine, then run
these commands inside it. In this repository, first run
`cd standalone_ocr_eval`. Install into the existing Conda environment:

```bash
conda run --no-capture-output -n myenv3.13 python -m pip install --no-deps --no-build-isolation .
```

For one page, pass two or more files containing OCR text for that *same* page:

```bash
conda run --no-capture-output -n myenv3.13 python -m ocr_eval page examples/page-a.md examples/page-b.md
conda run --no-capture-output -n myenv3.13 python -m ocr_eval page examples/page-a.md examples/page-b.md --ontology examples/ontology.json
```

The CLI prints JSON with the overall score, scores in file order, pairwise
diagnostics, and settings. Files may be `.txt`, `.md`, `.markdown`, `.html`,
`.htm`, or supported single-page OCR `.json`. Multi-page JSON is rejected.

For page-aligned batches, write a JSON file like:

```json
{
  "invoice": {
    "engine_a": {"pages": ["Page one text", "Page two text"], "input_format": "plain"},
    "engine_b": {"pages": ["Page one text", "Page two text"], "input_format": "plain"},
    "engine_c": {"pages": ["Page one text", "Page two text"], "input_format": "plain"}
  }
}
```

At least three methods with equal page counts are required for batch ranking.
`input_format` may be `plain`, `markdown`, or `html`; it defaults to `markdown`.

```bash
conda run --no-capture-output -n myenv3.13 python -m ocr_eval batch examples/aligned.json --output results
conda run --no-capture-output -n myenv3.13 python -m ocr_eval batch examples/aligned.json --ontology examples/ontology.json --entity-weight 3
```

The batch command writes `results.json`, `ranking.csv`,
`page_differences.json`, `text_volume.json`, and `normalized_pages.json`.
Use `--weighting page` to give every page equal weight across documents;
the default gives each document equal weight. `--overlap-weight` controls the
blend with sequence agreement; the default `1` scores regions with an ontology, or token overlap without one.

```python
import json
from pathlib import Path
from ocr_eval import score_page, score_models, score_page_details, evaluate

score = score_page(["examples/page-a.md", "examples/page-b.md"])
region_score = score_page(["examples/page-a.md", "examples/page-b.md"], ontology_path="examples/ontology.json")
details = score_page_details(["examples/page-a.md", "examples/page-b.md"])
batch_data = json.loads(Path("examples/aligned.json").read_text(encoding="utf-8"))
ranking, pairs, volumes, normalized = evaluate(batch_data)
```

With an ontology, the default primary metric is **region agreement**. Locate lexical
ontology anchors and select one window with 16 surrounding tokens on each side.
Algorithm 6.1 uses anchor-relative boundaries, without fixed-size chunks. Each
token and bigram shares weight equally among covering windows, so overlapping
windows do not duplicate selected mass. Canonical
anchor tags normalize aliases and CamelCase/spaced forms. Common function-word
labels and labels shorter than three characters are suppressed.

Regions with the same central anchor tag are matched one-to-one by maximum weighted
agreement. Ambiguous anchors require identical window tokens, a shared unambiguous
neighbouring anchor, or at least two distinct shared context words with word Dice
at least 0.5. Context words exclude anchor tokens, numbers, common function words
and words shorter than three characters. Generic labels alone cannot align unrelated
passages. Each match uses equal overlap-weighted token Dice and ordered-bigram Dice, weighted by
combined region token mass. Unmatched regions receive zero. All regional text and
numbers contribute; no exact field/value parsing occurs. Ambiguous owners do not
incur an automatic penalty; matching requires contextual evidence. Changing numbers
or local order lowers agreement. Rejected context candidates are counted and up to
20 examples per pair are saved. Regenerate algorithm 6.0 metrics before ranking.

Use `region_context_tokens=16` / `--region-context-tokens 16` (integer 1–128) to
change the window. `ontology_region_agreement` is the primary metric;
`ontology_anchor_agreement` and per-output `localization` are separate diagnostics.
Coverage records selected/total tokens, selected fraction, region/anchor counts
and ambiguous anchors. Pair diagnostics contain selected passages, character/line
boundaries, anchor evidence, matches and unmatched regions. The exported functions
`extract_ontology_regions(text, ontology_path, input_format, context_tokens)` and
`region_metrics(left_regions, right_regions)` support direct inspection.

Pages with no located regions across peers are excluded; entirely unscorable
inputs raise an error. Peers without regions receive zero if others have regions.
Dense ontologies may select most of the text; missing or misspelled labels may
prevent localization. Labels without values can agree. Inspect coverage before
interpreting scores. These are text passages, with no spatial coordinates.
Consensus does not establish field accuracy or ontology quality.

`ontology_mode="values"` / `--ontology-mode values` retains exact field scoring.
`ontology_mode="mentions"` / `--ontology-mode mentions` selects label-weighted
token overlap; `entity_weight` affects that mode only.

An ontology must contain a nonempty `concepts` object. Each concept key maps
to an object with optional `name`, `children`, and `fields`; the last two are
lists of strings. Optional `aliases`, `value_type` and `field_types` control
canonical label matching and (in values mode) typed value comparison. For example:

```json
{"concepts": {"Invoice": {"fields": ["IBAN", "Total"], "field_types": {"IBAN": "identifier", "Total": "amount"}}}}
```

Pass `ontology_path` or `--ontology` to use a different file on any run. A
supplied missing or invalid file raises an error. The parent repository's
`document_ontology.json` is one possible input, but this folder and its wheel
do not require or contain it.


In the optional **values mode**, values are extracted from explicit label/value pairs, typed inline values and
adjacent label/value lines, plus Markdown, HTML and tab-separated tables. Lines
and cells are retained in normalized batch exports so replay keeps associations.
CamelCase/spaced labels and declared aliases share canonical fields. Ambiguous
field owners are reported; section headings can supply the owning concept.

Agreement is `2 * matching_valid_records / total_records`. Each record is bound
to its canonical field, stable unlabelled first-column table key (when available)
and occurrence. Invalid/missing values and extraction issues receive no match
credit. Repeated fields without stable keys use source occurrence order. Pages
without field evidence are excluded; entirely unscorable inputs raise an error.
Two peers without fields receive zero when another peer has field evidence.

Supported `value_type` / `field_types` values: `text`, `identifier`, `number`,
`amount`, `date`, `currency`. Common names infer a type; use overrides for labels
such as `Total`. Identifiers retain leading zeroes; standard identifier labels
normalize spaces/case. Amounts preserve signs, units and associated currencies,
using exact Decimal comparison with dot decimals and grouped thousands commas.
Accounting parentheses denote negative amounts in field scoring; token diagnostics
still preserve them literally. Dates normalize only when unambiguous; optional
`date_order: "dmy"` or `"mdy"` on the ontology or owning concept resolves day/month
ambiguity. Text values normalize Unicode, whitespace and case.

Pair diagnostics include `ontology_field_value_agreement`, `per_field_agreement`,
extracted and unmatched records with source evidence, invalid values and ambiguous
label/column-count issues. `extract_field_values(text, ontology_path, input_format)`
and `field_value_metrics(left_extraction, right_extraction)` are also exported.
Unlabelled values, undeclared synonyms and misspelled labels are not inferred.
These scores measure consensus among extracted values; inspect extraction coverage
and use independently reviewed references when measuring accuracy.
