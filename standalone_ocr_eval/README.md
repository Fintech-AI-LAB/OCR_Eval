# OCR text evaluation library

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
