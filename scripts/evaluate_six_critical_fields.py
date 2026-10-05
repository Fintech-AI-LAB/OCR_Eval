"""Reproduce the exploratory critical-field pilot on the six saved OCR outputs.

Run in myenv3.13. This is a corpus-specific, in-sample extraction experiment,
not a general extractor or a held-out estimate of OCR model performance.
References were transcribed by Codex from seven rendered source pages.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'standalone_ocr_eval'))
from ocr_eval import evaluate_benchmark, consensus_family_sensitivity
from ocr_eval.field_values import field_text
from ocr_eval.profiles import fingerprint

OUT = ROOT / 'output/six_critical_field_evaluation'
SAVED = ROOT / 'output/ontology_region_comparison_v6_1/six_documents'
FAMILIES = {'mistral': 'mistral', 'openai-6': 'tesseract', 'fable-5-1': 'tesseract',
            'mineru': 'mineru', 'insavlo': 'unknown-insavlo', 'deepseek': 'deepseek'}
SOURCES = {
    'csa': '1000759706 SSGA - SCB - CSA - EXECUTED - ANON (CLEAN).pdf',
    'board': 'board_resolution.pdf',
    'trade': 'Trade-1118348-260625.001369.01.01.tif0.pdf',
}
SOURCE_SHA256 = {
    'board': '086d8b746bf9695cd44bb947dbfbb0ed97d4ceed43767a57a060558c45117e34',
    'csa': 'ec38d587b7baffadf38e5311d91dff4e99b5af239df95fa334807b9e82c25b3f',
    'trade': '923126acb7ade6d30f586c73b5b9d30d076629b4d6544629f1d15afaf5b2d2cd',
}
# Diagnostic observations made by inspecting the decoded OCR in the relevant
# section. These do not amend predictions or grant any additional score credit.
RETAINED_BUT_UNEXTRACTED = {
    ('mistral', 'board', 'ResolutionExportAccount'): 'Value retained beside USD Expert Proceeds Account; label misread.',
    ('mineru', 'board', 'ResolutionDomiciliaryAccount'): 'Value retained beside USD Nominalary Account; label misread.',
    ('deepseek', 'board', 'ResolutionExportAccount'): 'Value retained beside USD Export Process Account; label misread.',
    ('fable-5-1', 'board', 'ResolutionCurrentAccount'): 'Value retained after Account, but reading order separates Current from Account.',
    **{(m, 'trade', f): 'Value retained in the first invoice, but the DOC NO and DATE labels are missing.'
       for m in ['openai-6', 'fable-5-1'] for f in ['InvoiceNumber', 'InvoiceDate']},
}
# Evidence rectangles are approximate, in the displayed PDF orientation. They
# locate source evidence for review; this pilot does not evaluate localization.
GOLD = {
    'board': [
        ('LetterDate', 'September 13, 2019', 'date', 1, [.16,.17,.30,.20]),
        ('LetterCurrentAccount', '5202425205', 'identifier', 1, [.17,.33,.72,.40]),
        ('LetterDomiciliaryAccount', '7302473268', 'identifier', 1, [.17,.33,.72,.40]),
        ('LetterExportAccount', '0002473316', 'identifier', 1, [.17,.33,.72,.40]),
        ('LetterFXBidAccount', '3102768313', 'identifier', 1, [.17,.33,.72,.40]),
        ('LetterRemovedSignatory', 'Amit Agarwal', 'text', 1, [.19,.38,.38,.41]),
        ('LetterAddedSignatory', 'Satya Narayan Patra', 'text', 1, [.20,.45,.40,.47]),
        ('ResolutionCurrentAccount', '0502425205', 'identifier', 3, [.14,.20,.70,.25]),
        ('ResolutionDomiciliaryAccount', '8002473268', 'identifier', 3, [.14,.20,.70,.25]),
        ('ResolutionExportAccount', '3302473316', 'identifier', 3, [.14,.20,.70,.25]),
        ('ResolutionFXBidAccount', '0002768331', 'identifier', 3, [.14,.20,.70,.25]),
        ('ScheduleEmail', 'finance@africanindustries.com', 'text', 4, [.18,.35,.42,.38]),
        ('ScheduleSigningRule', 'TWO SIGNATORIES ONE FROM EACH CATEGORY (A & B) MUST SIGN JOINTLY',
         'text', 4, [.18,.54,.68,.59]),
    ],
    'csa': [
        ('AgreementDate', '7 March 2017', 'date', 1, [.20,.15,.80,.50]),
        ('Valuation5To10Years', '95%', 'text', 9, [.1,.05,.9,.35]),
        ('Valuation10To30Years', '90%', 'text', 9, [.1,.05,.9,.35]),
        ('Valuation30To50Years', '87%', 'text', 9, [.1,.05,.9,.35]),
        ('RoundingCurrency', 'EUR', 'currency', 9, [.1,.35,.9,.85]),
        ('RoundingAmount', '10,000', 'amount', 9, [.1,.35,.9,.85]),
    ],
    'trade': [
        ('CoverDealNumber', '958152731828', 'identifier', 1, [.05,.05,.95,.55]),
        ('CoverCustomerName', 'EXXONMOBIL ASIA PACIFIC PTE', 'text', 1, [.05,.05,.95,.55]),
        ('CoverCustomerID', '8354618', 'identifier', 1, [.05,.05,.95,.55]),
        ('CoverCurrency', 'USD', 'currency', 1, [.05,.05,.95,.55]),
        ('CoverAmount', '38,973.63', 'amount', 1, [.05,.05,.95,.55]),
        ('InvoiceNumber', '32310062', 'identifier', 9, [0,0,1,1]),
        ('InvoiceDate', '12.06.2025', 'text', 9, [0,0,1,1]),
        ('InvoiceCustomerAccount', '163511', 'identifier', 9, [0,0,1,1]),
    ],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, obj):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def annotations():
    profiles = {'schema_version': 1, 'profiles': {}}
    refs = {'schema_version': 1, 'annotation_status': 'verified', 'documents': {}}
    questions = {
        'board': 'Which accounts, signatories and signing instructions belong to each mandate section?',
        'csa': 'Which agreement date, maturity-specific valuation percentages and rounding amount apply?',
        'trade': 'Which customer, deal, amount and invoice identifiers belong to the cover and first invoice?',
    }
    for doc, rows in GOLD.items():
        fields, values = {}, {}
        for field, value, kind, page, box in rows:
            fields[field] = {'value_type': kind, 'critical': True, 'required': True, 'weight': 1,
                             'rationale': 'Selected for the pilot task: ' + questions[doc] +
                             ' Uniform weight is a provisional evaluation policy, not an estimated business cost.'}
            if 'Account' in field or field.endswith(('ID', 'Number')):
                fields[field]['constraints'] = {'pattern': '[0-9]+'}
            if field == 'InvoiceDate':
                fields[field]['comparison'] = 'literal'
            values[field] = {'status': 'present', 'value': value, 'page': page, 'bbox': box}
        if doc in {'csa', 'trade'}:
            prefix = 'Rounding' if doc == 'csa' else 'Cover'
            fields[prefix + 'Amount']['requires'] = [prefix + 'Currency']
        profiles['profiles'][doc] = {
            'review_status': 'reviewed',
            'reviewed_by': 'Codex technical review for this exploratory pilot; no business-owner approval or cost elicitation.',
            'competency_questions': [questions[doc]], 'fields': fields,
        }
        path = ROOT / 'data' / SOURCES[doc]
        if sha(path) != SOURCE_SHA256[doc]:
            raise ValueError(f'{doc}: reference PDF changed; visually reannotate before scoring.')
        refs['documents'][doc] = {
            'profile': doc, 'template_id': doc + '-pilot', 'split': 'validation',
            'reviewed_by': 'Codex visual inspection of current rendered PDF pages, 2026-10-05; not independently human-adjudicated.',
            'source': {'path': str(path), 'sha256': sha(path)},
            'records': [{'id': 'document', 'fields': values}],
        }
    return profiles, refs


def extract(doc, text):
    """Same label/section rules for every method; never consumes gold values.

    Strict labels deliberately expose label loss and reading-order failure.
    Captures retain OCR value errors and offsets into the saved decoded text.
    """
    found = []

    def get(field, pattern, start=0, end=None, group='value'):
        match = re.search(pattern, text[start:end], re.I | re.S)
        if match:
            a, b = match.span(group)
            a, b = start + a, start + b
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b-1].isspace():
                b -= 1
            if a < b:
                found.append({'record': 'document', 'field': field, 'value': text[a:b],
                              'decoded_span': [a, b], 'extraction_rule': pattern})

    if doc == 'board':
        letter_end = re.search(r'We attached the following', text, re.I)
        letter_end = letter_end.start() if letter_end else 0
        get('LetterDate', r'(?P<value>[A-Za-z]+\s+\d{1,2},\s+\d{4})', end=letter_end)
        resolution = re.search(r'That the company mandate', text, re.I)
        resolution_end = re.search(r'be changed as follows', text[resolution.start():], re.I) if resolution else None
        sections = [('Letter', 0, letter_end)]
        if resolution and resolution_end:
            sections.append(('Resolution', resolution.start(), resolution.start()+resolution_end.end()))
        for prefix, start, end in sections:
            for key, label in [('CurrentAccount', r'Current\s+Account'),
                               ('DomiciliaryAccount', r'USD\s+Domiciliary\s+Account'),
                               ('ExportAccount', r'USD\s+Export\s+Proceeds\s+Account'),
                               ('FXBidAccount', r'FX\s+Bid\s+Account')]:
                get(prefix+key, label+r'\s+(?P<value>[^\s,]+)', start, end)
        # The letter lists the removed and added signatory, in that order.
        names = list(re.finditer(r'Mr\.\s+(?P<value>[^\n\t]+?)(?=\s*[-–]\s*Category|\s+Category|\n|\t)',
                                 text[:letter_end], re.I))
        # Ignore salutation attention names: the mandate paragraph starts here.
        mandate = re.search(r'With reference', text[:letter_end], re.I)
        names = [m for m in names if mandate and m.start() > mandate.start()]
        for key, match in zip(['LetterRemovedSignatory', 'LetterAddedSignatory'], names):
            a, b = match.span('value')
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b-1].isspace():
                b -= 1
            found.append({'record': 'document', 'field': key, 'value': text[a:b],
                          'decoded_span': [a, b], 'extraction_rule': 'Ordered Mr. names in letter mandate section.'})
        nominees = list(re.finditer(r'NOMINE?ES', text, re.I))
        if nominees:
            start = nominees[-1].end()
            get('ScheduleEmail', r'Email\s*[:\t]\s*(?P<value>[^\n\t]*@[^\n\t]*?)(?=\s+Email|\n|\t|$)', start)
            get('ScheduleSigningRule', r'(?P<value>TWO\s+SIGNATORIES\s+ONE\s+FROM\s+EACH\s+CATEGORY\s*\([^)]*\)\s+MUST\s+SIGN\s+JOINTLY)', start)
    elif doc == 'csa':
        get('AgreementDate', r'dated as of\s+(?P<value>\d{1,2}\s+[A-Za-z]+\s+\d{4})', end=2000)
        for lower, upper in [(5, 10), (10, 30), (30, 50)]:
            # Value can precede the wrapped upper-bound phrase in plain OCR.
            get(f'Valuation{lower}To{upper}Years', rf'More than\s+{lower}\s+years\s+but not more'
                rf'(?=(?:(?!More than|Thresholds).){{0,250}}?than\s+{upper}\s+years)'
                rf'(?:(?!More than|Thresholds).){{0,250}}?(?P<value>\d+\s*%)')
        prefix = r'nearest integral multiple of\s+'
        get('RoundingCurrency', prefix+r'(?P<value>\S+)')
        get('RoundingAmount', prefix+r'\S+\s+(?P<value>[\d,]+(?:\.\d+)?)')
    elif doc == 'trade':
        cover_end = re.search(r'PRE-PROCESSING SHEET', text, re.I)
        cover_end = cover_end.start() if cover_end else 1500
        get('CoverDealNumber', r'Deal Number\s*/\s*Step Type\s+(?P<value>\S+)\s*/', end=cover_end)
        get('CoverCustomerName', r'Customer Name\s+(?P<value>.*?)\s*Customer ID', end=cover_end)
        get('CoverCustomerID', r'Customer ID\s+(?P<value>\S+)', end=cover_end)
        get('CoverCurrency', r'Currency and Amount\s+(?P<value>\S+)', end=cover_end)
        get('CoverAmount', r'Currency and Amount\s+\S+\s+(?P<value>[^\n\t]+)', end=cover_end)
        # First invoice only. No fallback searches for a matching reference value
        # in subsequent invoices, which would conceal missing or reordered data.
        bill_to = re.search(r'Bill to address', text, re.I)
        if bill_to:
            start = max(0, bill_to.start()-700)
            get('InvoiceNumber', r'DOC\s+NO\.?\s*:\s*(?P<value>\S+)', start, bill_to.start())
            get('InvoiceDate', r'(?<!\w)DATE\s*:\s*(?P<value>\S+)', start, bill_to.start())
            # Stop before another invoice header/address, even when the first
            # invoice omitted a field that a subsequent copy happens to retain.
            after = text[bill_to.end():]
            next_invoice = re.search(r'DOC\s+NO\.?\s*:|Bill to address|COMMERCIAL INVOICE', after, re.I)
            end = min(bill_to.end()+1800, bill_to.end()+next_invoice.start()) if next_invoice else bill_to.end()+1800
            get('InvoiceCustomerAccount', r'Your customer account\s*:\s*(?P<value>\S+)', bill_to.end(), end)
    return found


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    input_path = SAVED / 'input.json'
    data = json.loads(input_path.read_text())
    prior = json.loads((SAVED / 'with_ontology/results.json').read_text())
    provenance = prior['provenance']
    assert sha(input_path) == provenance['filtered_input_sha256'], 'Frozen input changed.'
    verified_artifacts = {}
    for record in provenance['artifacts']:
        for path, expected in record.get('artifact_sha256', {}).items():
            actual = sha(Path(path))
            assert actual == expected, f'OCR artifact changed: {path}'
            verified_artifacts[path] = actual
    profiles, refs = annotations()
    predictions = {'schema_version': 1, 'profile_sha256': fingerprint(profiles), 'methods': {}}
    for method, family in FAMILIES.items():
        docs = {}
        for doc in GOLD:
            raw = data[doc][method]
            text = field_text('\n\n'.join(raw['pages']), 'markdown')
            folder = OUT / 'decoded' / method
            folder.mkdir(parents=True, exist_ok=True)
            (folder / (doc+'.txt')).write_text(text)
            docs[doc] = {'fields': extract(doc, text)}
            for item in docs[doc]['fields']:
                a, b = item['decoded_span']
                assert item['value'] == text[a:b], 'Prediction must retain the captured OCR value.'
        predictions['methods'][method] = {
            'family': family, 'documents': docs,
            'provenance': {'saved_label': method, 'source_records': [r for r in provenance['artifacts'] if r['engine_label'] == method],
                           'extraction': 'Shared deterministic section/label rules; developed on these documents; no value repair.'},
        }
    result = evaluate_benchmark(profiles, refs, predictions, split='validation')
    result['meaning'] = ('Exploratory in-sample OCR-plus-extraction pilot on 27 selected fields from seven source pages. '
                         'Codex image-checked references, not independently human-adjudicated or a held-out benchmark.')
    result['pilot_limitations'] = {
        'selection': 'Convenience sample informed by earlier spot checks; no claim of exhaustive critical-field identification.',
        'weights': 'Uniform per-field weights within each document; equal document weights. Business costs not elicited.',
        'localization': 'Not assessed: predictions have no comparable boxes. Ignore zero localization credit in generic metrics.',
        'ocr_only_track': 'Not assessed: no fixed-region OCR reruns or transcripts.',
        'criticality_review': 'Technical pilot selection only; no business-owner approval.',
    }
    for name, value in [('profiles.json', profiles), ('references.json', refs), ('predictions.json', predictions),
                        ('results.json', result), ('families.json', FAMILIES),
                        ('run_manifest.json', {'input_path': str(input_path), 'input_sha256': sha(input_path),
                         'script_sha256': sha(Path(__file__)), 'verified_artifact_count': len(verified_artifacts),
                         'verified_artifacts': verified_artifacts})]:
        write(name, value)
    comparisons = {}
    for mode in ['with_ontology', 'without_ontology']:
        pairs = json.loads((SAVED / mode / 'pair_differences.json').read_text())
        # These are three whole-document units; page=1 is only a matrix index.
        assert len(pairs) == 45
        sensitivity = consensus_family_sensitivity(pairs, FAMILIES)
        write(mode + '_family_sensitivity.json', sensitivity)
        original = json.loads((SAVED / mode / 'results.json').read_text())
        comparisons[mode] = {'ranking': original['ranking'], 'family_balanced_scores': sensitivity['family_balanced_scores'],
                             'source_results': str(SAVED / mode / 'results.json'),
                             'source_results_sha256': sha(SAVED / mode / 'results.json')}
    write('consensus_comparison.json', comparisons)
    rows = []
    for rank in result['ranking']:
        method = rank['method']; r = result['methods'][method]
        rows.append({**rank, 'correct': r['metrics']['critical_correct'], 'total': r['metrics']['critical_fields'],
                     **{doc+'_correct': x['counts']['critical_correct'] for doc,x in r['documents'].items()},
                     'missing': r['metrics']['missing_fields'], 'wrong': r['metrics']['wrong_values']})
    with (OUT / 'ranking.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    audit = []
    for method, r in result['methods'].items():
        for doc, scored in r['documents'].items():
            text = (OUT / 'decoded' / method / (doc+'.txt')).read_text()
            for x in scored['field_results']:
                note = RETAINED_BUT_UNEXTRACTED.get((method, doc, x['field']), '')
                if note:
                    assert x['reason'] == 'missing' and x['reference_value'] in text
                audit.append({'method': method, 'document': doc, 'field': x['field'],
                              'source_page': x['reference_location']['page'],
                              'reference': x['reference_value'], 'prediction': x['predicted_value'],
                              'result': x['reason'], 'review_note': note})
    assert len(audit) == 162 and all(r['total'] == 27 for r in rows)
    assert sum(bool(x['review_note']) for x in audit) == 8
    with (OUT / 'field_audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0])); writer.writeheader(); writer.writerows(audit)
    make_report(result, rows, comparisons, audit, len(verified_artifacts))
    print(json.dumps(rows, indent=2))
    print('Incorrect or unextracted fields:')
    for method, r in result['methods'].items():
        for doc, scored in r['documents'].items():
            for x in scored['field_results']:
                if not x['correct']:
                    print(method, doc, x['field'], repr(x['predicted_value']), x['reason'])
    print(f'Verified {len(verified_artifacts)} saved artifact hashes. Wrote {OUT}')


def make_report(result, rows, comparisons, audit, artifact_count):
    lines = ['# Six saved OCR outputs: critical-field pilot', '', result['meaning'], '',
        'This run evaluates saved text; it does not run new OCR inference. The main score measures the '
        'saved OCR output plus a shared, strict field extractor. It is sensitive to labels and reading order, '
        'as well as the field values themselves. It is not an OCR-only character accuracy score.', '',
        '| Saved output | Score / 100 | Correct / 27 | Board / 13 | CSA / 6 | Trade / 8 | Unextracted | Wrong value/binding |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['critical_field_score']:.2f} | {r['correct']}/27 | "
                     f"{r['board_correct']}/13 | {r['csa_correct']}/6 | {r['trade_correct']}/8 | {r['missing']} | {r['wrong']} |")
    lines += ['', '**Scoring:** each selected field has weight 1; scores are averaged within each document, '
              'then the three document scores are averaged equally. Thus the score is not simply correct / 27. '
              'Missing fields stay in the denominator. Amounts require the associated currency to be correct. '
              'The two tested amount/currency relationships are correct for all six outputs. Account fields are '
              'bound to either the letter or resolution; the values in those two source sections differ and must '
              'not be substituted for one another. Invoice date uses literal DD.MM.YYYY comparison. Other values '
              'use the profile type normalization, with raw values retained.', '',
              'The selected task fields cover: board letter date, four accounts, removed/added signatories, '
              'four resolution accounts, schedule email and signing rule; CSA agreement date, three maturity-specific '
              'valuation percentages, rounding amount/currency; trade cover deal/customer identifiers, customer name '
              'and amount/currency, plus first-invoice number/date/customer account.', '',
              '## Failure analysis', '',
              'Eight of the 15 unextracted fields retain the correct value in the relevant OCR text. The strict '
              'extractor cannot bind them because labels or reading order are damaged. These are recorded below '
              'without changing their scores. This is a material limitation of comparing OCR engines through a '
              'strict parser; it must not be reported as eight missing or incorrect value transcriptions.', '',
              '| Output | Field | Diagnosis |', '|---|---|---|']
    for x in audit:
        if x['review_note']:
            lines.append(f"| {x['method']} | {x['field']} | {x['review_note']} |")
    lines += ['', 'The remaining failures are:', '',
              '- **Fable/Tesseract:** the letter current account starts with `§` instead of `5`; the resolution '
              'FX account changes a digit; the email and `(A & B)` signing rule are corrupted. Broken reading '
              'order splits the domiciliary account and causes the extractor to bind `respectively` as its value. '
              'That last case is a binding failure, not evidence that the OCR transcribed the number as that word.',
              '- **MinerU:** the board schedule output stops after its header/client information; the selected '
              'email and signing rule are unavailable.',
              '- **DeepSeek:** the board schedule fields and the selected first-invoice fields are unavailable '
              'to this extractor. The cover customer name reads `EXONMOBIL` instead of `EXXONMOBIL`.',
              '- **Insavlo:** all selected fields pass. This does not establish its accuracy outside the sample '
              'or verify its model identity/source provenance.', '',
              'All six pass the six selected CSA fields, so this small CSA sample does not distinguish them.', '',
              '## Full-document agreement and family sensitivity', '',
              'The agreement results below reuse the existing six-method, three-whole-document pair matrices. '
              'They are peer consensus, not ground-truth accuracy. Family-balanced scores are newly calculated '
              'by excluding same-family peers and averaging the remaining engine families equally. The two '
              'Tesseract-based outputs share one family. Insavlo is provisionally separate because its engine '
              'is unknown; independence is not established.', '',
              '| Output | Ontology-region agreement | Family-balanced agreement | Whole-text token agreement |',
              '|---|---:|---:|---:|']
    ont = {r['model']: r['consensus_score'] for r in comparisons['with_ontology']['ranking']}
    tok = {r['model']: r['consensus_score'] for r in comparisons['without_ontology']['ranking']}
    fam = comparisons['with_ontology']['family_balanced_scores']
    for r in comparisons['with_ontology']['ranking']:
        m = r['model']
        lines.append(f'| {m} | {ont[m]:.2f} | {fam[m]:.2f} | {tok[m]:.2f} |')
    lines += ['', 'Mistral has the highest whole-document agreement under both original and family-balanced '
              'weighting. That does not conflict with Insavlo having the highest score in this selected-field '
              'pilot: they measure different outcomes. Leave-one-family-out results are saved separately.', '',
              '## Scope and provenance', '',
              '- 27 fields across seven inspected source pages: board 1/3/4, CSA 1/9, trade 1/9; 162 method-field comparisons. '
              'The underlying sources contain 48 physical pages. The other 41 pages and unselected fields are not covered by reference accuracy.',
              '- References were visually checked by Codex against rendered current PDFs. They have not been independently '
              'human-adjudicated. Selection was informed by earlier spot checks; this is a convenience sample.',
              '- Profiles were technically reviewed for this pilot. Equal field weights are provisional; domain experts '
              'have not approved criticality or estimated error costs. No exhaustive ontology-based field selection is claimed.',
              '- Extraction rules were developed on these same documents. References are marked `validation`, not `test`; '
              'there is no held-out estimate, statistical significance test or general best-engine claim.',
              '- `openai-6` and `fable-5-1` are saved directory labels for Tesseract-based artifacts, not verified models '
              'with those names. OpenAI-labelled text includes selective review. Fable uses its raw page-text layer; '
              'its separately reviewed field layer is not included. Therefore these are comparisons of saved artifacts, '
              'not controlled runs of six independent engines.',
              '- Fable and Insavlo have no verified source hashes. Mistral trade came from an earlier TIFF whose pixel '
              'equivalence to the current reference PDF is unverified. DeepSeek trade page 10 was truncated; this page '
              'is outside the selected reference fields but affects full-document agreement.',
              f'- Verified the frozen input hash and {artifact_count} original saved OCR artifact hashes. Source PDF hashes, '
              'full upstream provenance, predictions, matching rules and decoded text offsets are preserved.',
              '- Comparable prediction boxes are unavailable. Localization and fixed-region OCR-only CER are not evaluated; '
              'ignore generic zero localization credit in `results.json`. Reference boxes are approximate review regions, '
              'including whole-page evidence boxes for the rotated invoice.', '',
              '## Reproduce and inspect', '',
              '```bash', 'conda run --no-capture-output -n myenv3.13 python scripts/evaluate_six_critical_fields.py', '```', '',
              'Outputs in this directory: `ranking.csv`, `field_audit.csv` (all 162 comparisons), `results.json`, '
              '`profiles.json`, `references.json`, `predictions.json`, `run_manifest.json`, `decoded/`, '
              '`consensus_comparison.json` and the two family-sensitivity files. No source OCR artifacts are modified.', '',
              'The next experiment should freeze a reviewed field set, annotate a larger held-out sample, and compare '
              'strict extraction with layout-aware extraction and fixed-region transcription on the same references. '
              'These results already show why label damage must be separated from value transcription errors.', '']
    (OUT / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
