#!/usr/bin/env python3
"""Export page-aligned JSON/Markdown and a local three-column comparison."""
import argparse
import hashlib
import html
import json
from pathlib import Path

from ocr_common import ROOT, write_json
from run_astra_ocr import DEFAULT_FILES
from run_mistral_ocr import markdown as mistral_markdown

ENGINES = ('apple_vision', 'mistral', 'mineru')


def compatible(run, source):
    metadata = json.loads((run / 'source.json').read_text())
    content = source.read_bytes()
    if 'source_sha256' in metadata:
        return metadata['source_sha256'] == hashlib.sha256(content).hexdigest()
    # Validate hashes used by the shared runner and the earlier Mistral runner.
    config = {k: v for k, v in metadata.items() if k not in ('source', 'responses', 'requests')}
    digest = hashlib.sha256(content)
    digest.update(json.dumps(config, sort_keys=True).encode())
    legacy = hashlib.sha256(content)
    legacy.update(metadata.get('model', '').encode())
    return run.name in (digest.hexdigest()[:16], legacy.hexdigest()[:16])


def normalize(engine, run):
    metadata = json.loads((run / 'source.json').read_text())
    if engine in ('astra', 'apple_vision'):
        return json.loads((run / 'document.json').read_text())['pages']
    result = []
    for path in sorted(run.glob('response-*.json')):
        raw = json.loads(path.read_text())
        if engine == 'mistral':
            for page in sorted(raw.get('pages', []), key=lambda p: p.get('index', 0)):
                result.append({'page_number': len(result) + 1,
                               'markdown': mistral_markdown({'pages': [page]})})
        else:
            # Use the same official conversion as the MinerU runner; preserve raw blocks.
            from mineru_vl_utils.post_process import json2md
            result.append({'page_number': len(result) + 1,
                           'markdown': json2md(raw), 'blocks': raw})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'comparison')
    args = parser.parse_args()
    sources = args.input or [ROOT / 'data' / name for name in DEFAULT_FILES]
    reports = []
    sections = []
    for source in sources:
        source = source.resolve()
        if not source.is_file():
            parser.error(f'Missing input: {source}')
        target = args.output / source.name
        target.mkdir(parents=True, exist_ok=True)
        engines = {}
        for engine in ENGINES:
            candidates = [p.parent for p in (ROOT / 'output' / engine / source.name).glob('*/source.json')
                          if compatible(p.parent, source)]
            if not candidates:
                engines[engine] = {'status': 'missing', 'pages': []}
                continue
            run = max(candidates, key=lambda p: (p / 'source.json').stat().st_mtime_ns)
            metadata = json.loads((run / 'source.json').read_text())
            try:
                normalized = normalize(engine, run)
            except ImportError:
                engines[engine] = {'status': 'converter dependency missing', 'run': str(run), 'pages': []}
                continue
            record = {'schema_version': 1, 'source': source.name,
                      'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                      'backend': engine, 'model': metadata.get('model'),
                      'run': str(run), 'page_count': len(normalized), 'pages': normalized}
            write_json(target / f'{engine}.json', record)
            (target / f'{engine}.md').write_text('\n\n'.join(
                f'<!-- page:{p["page_number"]} -->\n\n{p["markdown"]}' for p in normalized
            ), encoding='utf-8')
            engines[engine] = {'status': 'available', **record}
        import pypdfium2 as pdfium
        with pdfium.PdfDocument(source) as pdf:
            expected = len(pdf)
        reports.append({'source': source.name, 'expected_page_count': expected,
                        'engines': {e: {k: v for k, v in data.items() if k != 'pages'}
                                    for e, data in engines.items()}})
        sections.append(f'<h2>{html.escape(source.name)}</h2>')
        for index in range(max(expected, *(len(x['pages']) for x in engines.values()))):
            sections.append(f'<h3>Physical page {index + 1}</h3><div class="grid">')
            for engine, record in engines.items():
                page = record['pages'][index] if index < len(record['pages']) else None
                text = page['markdown'] if page else f'[{record["status"]}: no result for this page]'
                sections.append(f'<section><h4>{engine}</h4><pre>{html.escape(text)}</pre></section>')
            sections.append('</div>')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'index.json', reports)
    (args.output / 'index.html').write_text('''<!doctype html><meta charset="utf-8">
<title>OCR comparison</title><style>
body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#172033}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
section{background:white;border:1px solid #ccd4df;border-radius:8px;padding:14px}
h4{margin:0 0 12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 monospace;max-height:65vh;overflow:auto}
</style><h1>OCR comparison</h1><p>Physical pages aligned across engines.
Verbatim Markdown is escaped for safe inspection, including HTML table markup.
Missing results are not empty transcriptions. No ground-truth accuracy is implied.
The most recently completed run matching the exact source bytes is selected.</p>'''
        + '\n'.join(sections), encoding='utf-8')
    print(f'Comparison: {args.output / "index.html"}')


if __name__ == '__main__':
    main()
