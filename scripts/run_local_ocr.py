#!/usr/bin/env python3
"""Local Apple Vision OCR; preserves line text, confidence and page coordinates."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from ocr_common import ROOT, destination_for, pages, write_json
from run_astra_ocr import DEFAULT_FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+')
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()
    scratch = ROOT / 'tmp' / 'pdfs'
    scratch.mkdir(parents=True, exist_ok=True)
    executable = scratch / 'vision_ocr'
    subprocess.run(['swiftc', '-module-cache-path', '/tmp/ocr-swift-cache',
                    str(ROOT / 'scripts' / 'vision_ocr.swift'), '-o', str(executable)], check=True)
    config = {'backend': 'apple_vision', 'model': 'Apple Vision VNRecognizeTextRequest',
              'os_version': platform.mac_ver()[0], 'dpi': args.dpi,
              'recognition_level': 'accurate', 'language_correction': False,
              'languages': ['en-US'], 'schema_version': 1}
    for path in args.input or [ROOT / 'data' / name for name in DEFAULT_FILES]:
        path = path.resolve()
        destination = destination_for(path, path.parent, ROOT / 'output' / 'apple_vision', config)
        records = []
        for index, image in enumerate(pages(path, args.dpi), 1):
            checkpoint = destination / f'response-{index:04d}.json'
            if checkpoint.exists():
                record = json.loads(checkpoint.read_text())
            else:
                png = scratch / 'current-page.png'
                image.save(png)
                result = subprocess.run([str(executable), str(png)], capture_output=True, text=True, check=True)
                blocks = json.loads(result.stdout)
                record = {'page_number': index, 'width': image.width, 'height': image.height,
                          'blocks': blocks, 'markdown': '\n\n'.join(b['text'] for b in blocks),
                          'uncertainties': [f'Low confidence line {i+1}: {b["text"]}'
                                            for i, b in enumerate(blocks) if b['confidence'] < .5]}
                write_json(checkpoint, record)
            records.append(record)
            print(f'{path.name}: page {index}, {len(record["blocks"])} lines', flush=True)
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(destination / 'document.json', {
            **config, 'source': path.name, 'source_sha256': source_hash,
            'bbox_format': 'normalized [left, top, right, bottom], top-left origin',
            'limitations': 'Automatic line OCR. Table structure, signatures, redactions and reading order may need review. Confidence is engine-reported, not calibrated accuracy.',
            'page_count': len(records), 'pages': records})
        (destination / 'document.md').write_text('\n\n'.join(
            f'<!-- page:{p["page_number"]} -->\n\n{p["markdown"]}' for p in records), encoding='utf-8')
        write_json(destination / 'source.json', {**config, 'source': path.name,
                   'source_sha256': source_hash, 'responses': len(records)})
        print(f'Saved: {destination}', flush=True)


if __name__ == '__main__':
    main()
