#!/usr/bin/env python3
"""Compare saved OCR artifacts; no inference calls, no assumed ground truth."""
import collections
import difflib
import statistics
import unicodedata
import hashlib
import html
import itertools
import json
import re
from pathlib import Path
from html.parser import HTMLParser
if __package__:
    from .ocr_common import ROOT, write_json
else:
    from ocr_common import ROOT, write_json

ENGINES = ['mistral', 'openai-6', 'fable-5-1', 'mineru']
DOCS = {'csa': '1000759706 SSGA - SCB - CSA - EXECUTED - ANON (CLEAN)',
        'board': 'board_resolution', 'trade': 'Trade-1118348-260625.001369.01.01.tif0'}
OUT = ROOT / 'output' / 'performance'

class Plain(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]
    def handle_data(self, data): self.parts.append(data)
    def handle_starttag(self, tag, attrs): self.parts.append(' ')
    def handle_endtag(self, tag): self.parts.append(' ')

def clean(text):
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'(?m)^\s{0,3}#{1,6}\s+', '', text)
    text = re.sub(r'(?m)^\s*[-*+]\s+', '', text)
    text = re.sub(r'(?m)^\s*\|?[ :|-]+\|[ :|-]*$', '', text)
    text = text.replace('|', ' ').replace('**', '').replace('`', '')
    parser = Plain(); parser.feed(text)
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', ''.join(parser.parts))).strip()


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def numeric_strings(text):
    # Preserve punctuation and leading zeroes; these are strings, not inferred fields.
    return sorted(set(re.findall(r'(?<!\w)\d+(?:[.,:/-]\d+)*(?:%)?(?!\w)', text)))


def load():
    data={}; manifest=[]
    for doc,stem in DOCS.items():
        source=ROOT/'data'/(stem+'.pdf')
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        data[doc]={}
        for engine in ENGINES:
            base=ROOT/'output'/engine
            if engine in ['openai-6','fable-5-1']:
                path=next(base.glob(stem+'*.json')); raw=json.loads(path.read_text())
                if engine=='openai-6':
                    page_list=[p['markdown'] for p in raw['pages']]
                    provenance=raw['engine']; matching=raw['source_sha256']==source_hash
                    fields=None
                else:
                    page_list=['\n'.join(line['text'] for line in p['lines']) for p in raw['full_page_text']]
                    provenance=raw['metadata']; matching=None
                    fields=None
                paths=[path]
            else:
                folder=next(base.glob(stem+'*')); path=next(folder.glob('*/source.json'))
                meta=json.loads(path.read_text()); paths=sorted(path.parent.glob('response-*.json'))
                raw_responses=[json.loads(p.read_text()) for p in paths]
                page_list=[];fields=None
                if engine=='mistral':
                    for response in raw_responses:
                        for p in response['pages']:
                            text=p['markdown']
                            for table in p.get('tables',[]) or []:
                                text=text.replace(f'[{table["id"]}]({table["id"]})',table['content'])
                            page_list.append(text)
                else:
                    page_list=['\n\n'.join(b.get('content','') or '' for b in blocks) for blocks in raw_responses]
                config={k:v for k,v in meta.items() if k not in ['source','responses','requests']}
                digest=hashlib.sha256(source.read_bytes());digest.update(json.dumps(config,sort_keys=True).encode())
                legacy=hashlib.sha256(source.read_bytes());legacy.update(meta['model'].encode())
                matching=path.parent.name in [digest.hexdigest()[:16],legacy.hexdigest()[:16]]
                if doc=='trade' and engine=='mistral':matching=None # TIFF submitted as separate PNG frames
                provenance=meta
            data[doc][engine]={'pages':[clean(page) for page in page_list]}
            manifest.append({'document':doc,'engine_label':engine,'metadata':provenance,
                             'source_pdf':str(source),'source_sha256':source_hash,'source_hash_verified':matching,
                             'artifacts':[str(p) for p in paths], 'page_count':len(page_list)})
    return data,manifest

def compare_page(doc, index, a, b, left, right):
    x, y = tokens(left), tokens(right)
    cx, cy = collections.Counter(x), collections.Counter(y)
    overlap = sum((cx & cy).values())
    denom = len(x) + len(y)
    # Symmetrize SequenceMatcher because its tie-breaking may depend on argument order.
    forward = difflib.SequenceMatcher(None, x, y, autojunk=False)
    reverse = difflib.SequenceMatcher(None, y, x, autojunk=False)
    changes = []
    for tag, i, j, k, l in forward.get_opcodes():
        if tag != 'equal':
            changes.append({'kind': tag, 'left_tokens': x[i:j], 'right_tokens': y[k:l],
                            'left_offset': i, 'right_offset': k,
                            'left_context': ' '.join(x[max(0,i-5):min(len(x),j+5)]),
                            'right_context': ' '.join(y[max(0,k-5):min(len(y),l+5)])})
    nx, ny = set(numeric_strings(left)), set(numeric_strings(right))
    return {'document':doc, 'page':index, 'left':a, 'right':b,
            'left_tokens':len(x), 'right_tokens':len(y), 'shared_token_occurrences':overlap,
            'token_overlap':2*overlap/denom if denom else 1,
            'sequence_similarity':(forward.ratio()+reverse.ratio())/2,
            'only_left_numeric_strings':sorted(nx-ny), 'only_right_numeric_strings':sorted(ny-nx),
            'changes':changes}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data,manifest=load()
    write_json(OUT/'normalized_pages.json',data)
    write_json(OUT/'provenance.json',manifest)
    metrics=[]; comparisons=[]; page_metrics=[]
    for doc,engines in data.items():
        lengths={len(r['pages']) for r in engines.values()}
        if len(lengths)!=1:
            raise ValueError(f'Page counts differ for {doc}; align before comparing.')
        for engine,record in engines.items():
            page_list=record['pages']
            metrics.append({'document':doc,'engine':engine,'pages':len(page_list),
                            'text_characters':sum(len(x) for x in page_list),
                            'tokens':sum(len(tokens(x)) for x in page_list)})
            for i,text in enumerate(page_list,1):
                page_metrics.append({'document':doc,'page':i,'engine':engine,
                                     'characters':len(text),'tokens':len(tokens(text))})
        for a,b in itertools.combinations(ENGINES,2):
            for i,(left,right) in enumerate(zip(engines[a]['pages'],engines[b]['pages']),1):
                comparisons.append(compare_page(doc,i,a,b,left,right))
    summaries=[]
    for doc in [*DOCS,'all']:
        for a,b in itertools.combinations(ENGINES,2):
            selected=[r for r in comparisons if r['left']==a and r['right']==b
                      and (doc=='all' or r['document']==doc)]
            denom=sum(r['left_tokens']+r['right_tokens'] for r in selected)
            summaries.append({'document':doc,'left':a,'right':b,'pages':len(selected),
                              'token_overlap':2*sum(r['shared_token_occurrences'] for r in selected)/denom if denom else 1,
                              'mean_page_sequence_similarity':statistics.mean(r['sequence_similarity'] for r in selected)})
    review_pages=[]
    for doc in DOCS:
        for page in range(1,len(data[doc][ENGINES[0]]['pages'])+1):
            pairs=[r for r in comparisons if r['document']==doc and r['page']==page]
            review_pages.append({'document':doc,'page':page,
                                 'mean_pairwise_overlap':statistics.mean(r['token_overlap'] for r in pairs),
                                 'mean_pairwise_sequence_similarity':statistics.mean(r['sequence_similarity'] for r in pairs)})
    review_pages.sort(key=lambda r:r['mean_pairwise_overlap'])
    write_json(OUT/'metrics.json',metrics)
    write_json(OUT/'page_metrics.json',page_metrics)
    write_json(OUT/'agreement.json',[{k:v for k,v in r.items() if k!='changes'} for r in comparisons])
    write_json(OUT/'text_differences.json',comparisons)
    write_json(OUT/'pairwise_summary.json',summaries)
    write_json(OUT/'review_priority.json',review_pages)
    write_report(metrics,summaries,review_pages,data)
    write_view(data,comparisons)
    print('Generated text-only comparison: 12 document/output records, 288 page pairs.')
    for r in summaries:
        if r['document']=='all': print(r)


def write_report(metrics,summaries,review_pages,data):
    lines=['# OCR text comparison — no ground truth',
           '', 'This report compares the text in four saved output sets across three documents and 48 pages. It does not rank accuracy, identify a correct engine, or score document structure. More text and higher agreement do not establish correctness.',
           '', 'The earlier structure measures, reference-anchor scores, field-extraction comparisons, speed/cost discussion, and best-engine recommendation are superseded by this text-only analysis.',
           '', '## Text normalization', '',
           'The comparison removes HTML tags, Markdown headings/bullets, table separators, image links and emphasis markers while retaining text, including text inside table cells. Unicode is normalized with NFKC and whitespace is collapsed. The view preserves case and punctuation. Similarity uses case-insensitive word/number tokens and ignores punctuation; numeric-string differences separately retain punctuation and leading zeroes. Line wrapping, layout, table geometry, bounding boxes, confidence values and extraction schemas are not evaluated.',
           '', 'The plain-text layer is taken from Mistral page responses, MinerU block content, openai-6 page Markdown, and Fable full_page_text lines. Fable’s separate reviewed fields, summaries and confidence commentary are excluded. Commentary already embedded inside an engine’s page transcription is retained and can affect agreement.',
           '', '## Pairwise text agreement', '',
           'Neither side is a reference. Token overlap counts shared token occurrences while ignoring order: 2 × overlap / total tokens. The overall overlap aggregates page-aligned counts, so longer pages carry more weight. Sequence similarity averages both directions of a word-level SequenceMatcher comparison, then gives each page equal weight; it is order-sensitive and is not Levenshtein accuracy, CER or WER.',
           '', '| Pair | Token overlap, all 48 pages | Mean page sequence similarity |',
           '|---|---:|---:|']
    for r in summaries:
        if r['document']=='all':lines.append(f"| {r['left']} ↔ {r['right']} | {r['token_overlap']:.1%} | {r['mean_page_sequence_similarity']:.1%} |")
    lines += ['', 'Agreement can reflect shared errors, shared preprocessing or manual editing. It should not be used as an accuracy ranking.', '', '### Token overlap by document', '', '| Pair | Agreement PDF | Board resolution | Trade PDF |','|---|---:|---:|---:|']
    for a,b in itertools.combinations(ENGINES,2):
        values=[next(r['token_overlap'] for r in summaries if r['document']==d and r['left']==a and r['right']==b) for d in DOCS]
        lines.append(f'| {a} ↔ {b} | '+' | '.join(f'{v:.1%}' for v in values)+' |')
    lines += ['', '## Text volume', '', 'Character/token totals describe what each output contains. A shorter output may omit text; a longer output may repeat or add text. Neither conclusion can be established from length alone.', '', '| Output | Agreement characters | Board characters | Trade characters | Total tokens |','|---|---:|---:|---:|---:|']
    for e in ENGINES:
        records=[next(r for r in metrics if r['document']==d and r['engine']==e) for d in DOCS]
        lines.append(f'| {e} | '+' | '.join(f"{r['text_characters']:,}" for r in records)+f" | {sum(r['tokens'] for r in records):,} |")
    lines += ['', '## Concrete text differences', '',
              '- **Board page 4:** Mistral and openai-6 include `finance@africanindustries.com` and the joint-signing instruction. MinerU’s page text is limited to `SCHEDULE` and the top client/submission text. Fable’s raw page text contains variants such as `Tinanne@airicanindust ies. com` and `(A &8)`. These are differences between outputs, not a determination of the correct reading.',
              '- **Trade page 1:** Mistral and MinerU say `1st Time Scan`; openai-6 and Fable raw text say `ist Time Scan`.',
              '- **Trade page 18:** Mistral starts with `2816128`; MinerU starts with `2:5/6/28`. openai-6 contains `HEGOTIATED BY` and `STANDARD CHARTERED pany`; Fable raw text contains fragments such as `CLAN Cuda Tp`. The outputs diverge substantially; none is designated ground truth.',
              '- **Board account-number passages:** the saved texts differ on some leading digits. The numeric-string differences in the JSON and page view preserve leading zeroes instead of treating identifiers as numbers or reconciling them across pages.',
              '', '## Pages with greatest disagreement', '', 'Sorted by the mean token overlap across all six pairs on each page. This is a review queue, not a list of proven errors. A short page can produce a low score from only a few differing tokens.', '', '| Document | Page | Mean pairwise token overlap |','|---|---:|---:|']
    for r in review_pages[:12]:lines.append(f"| {r['document']} | {r['page']} | {r['mean_pairwise_overlap']:.1%} |")
    lines += ['', '## Interpretation limits', '',
              '- `openai-6` and `fable-5-1` are folder labels. Their metadata identifies Tesseract plus visual review, so this is not a verified GPT-6-versus-Fable inference benchmark.',
              '- Fable’s reviewed fields are excluded, but selective corrections already present in openai-6 page text remain. The saved workflows are not controlled, identical experiments.',
              '- Mistral’s trade input was the earlier TIFF; the other outputs correspond to the current PDF. Exact pixel equivalence is unverified. Fable has no recorded source hash. See provenance.json for the matching evidence.',
              '- Shared passages are not verified correct. Text found on only one side is described as “only left/right,” not automatically as hallucination or omission. Reordering can lower sequence similarity without changing the words.',
              '- The prior source spot-check files and report are archived in superseded/ and are not used by this evaluation.',
              '', '## Files and reproduction', '',
              '- [Consensus ranking and sensitivity analysis](ranking.md) — run scripts/rank_ocr.py after regenerating this report.',
              '- [Plain-text page view and differences](pages.html)',
              '- [Pairwise summary](pairwise_summary.json) · [Page scores](agreement.json)',
              '- [Text and numeric-string differences](text_differences.json)',
              '- [Review queue](review_priority.json) · [Page lengths](page_metrics.json)',
              '- [Plain-text pages](normalized_pages.json) · [Provenance](provenance.json)',
              '', 'Rebuild with `conda run --no-capture-output -n myenv3.13 python scripts/evaluate_outputs.py`. The report, view and metrics are generated together. Original OCR artifacts are unchanged; no inference calls are made.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def write_view(data,comparisons):
    sections=[]
    for doc,engines in data.items():
        sections.append(f'<h2>{html.escape(DOCS[doc])}</h2>')
        for n in range(len(engines[ENGINES[0]]['pages'])):
            sections.append(f'<details><summary>Page {n+1}</summary><div class="grid">')
            for e in ENGINES:
                sections.append(f'<section><h3>{e}</h3><pre>{html.escape(engines[e]["pages"][n])}</pre></section>')
            sections.append('</div>')
            for r in comparisons:
                if r['document']!=doc or r['page']!=n+1:continue
                sections.append(f'<details><summary>{r["left"]} ↔ {r["right"]}: overlap {r["token_overlap"]:.1%}</summary>')
                sections.append('<p>Numeric strings only on left: '+html.escape(', '.join(r['only_left_numeric_strings']))+'<br>Only on right: '+html.escape(', '.join(r['only_right_numeric_strings']))+'</p>')
                sections.append('<table><tr><th>Left differing tokens (with context)</th><th>Right differing tokens (with context)</th></tr>')
                for c in r['changes']:
                    sections.append('<tr><td>'+html.escape(c['left_context'])+'</td><td>'+html.escape(c['right_context'])+'</td></tr>')
                sections.append('</table></details>')
            sections.append('</details>')
    (OUT/'pages.html').write_text('<!doctype html><meta charset="utf-8"><title>OCR text comparison</title>\n<style>body{font:15px system-ui;margin:24px;color:#172033;background:#f6f7f9}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}section{background:white;border:1px solid #ccd3df;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:70vh;overflow:auto;font:12px/1.5 monospace}summary{padding:14px;cursor:pointer}h3{font-size:14px}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border:1px solid #ccd3df;padding:8px;vertical-align:top;overflow-wrap:anywhere}</style>\n<h1>OCR text comparison — no ground truth</h1><p>Plain transcription only: formatting and table markup removed; table text retained. No accuracy ranking or structure scoring. Fable raw page text is used, not reviewed fields. openai-6 and fable-5-1 metadata identify Tesseract workflows. Pairwise differences have no correct/reference side.</p>'+''.join(sections),encoding='utf-8')


if __name__=='__main__':main()
