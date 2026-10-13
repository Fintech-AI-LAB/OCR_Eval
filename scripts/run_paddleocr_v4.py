#!/usr/bin/env python3
"""Run English PP-OCRv4 locally in myenv3.13 and checkpoint every page."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

from ocr_common import ROOT, add_arguments, discover, pages, process_file, write_json

PACKAGES = ROOT / '.cache/paddleocr-packages'
MODELS = ('PP-OCRv4_mobile_det', 'en_PP-OCRv4_mobile_rec',
          'PP-LCNet_x1_0_doc_ori', 'PP-LCNet_x1_0_textline_ori')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_from_result(result, page, size, seconds):
    """Preserve the native JSON and text order, without correcting OCR values."""
    native = result.json['res']
    texts = native['rec_texts']
    if not (len(texts) == len(native['rec_scores']) == len(native['rec_polys'])):
        raise ValueError('PaddleOCR returned inconsistent text/score/polygon counts')
    return {'page': page, 'text': '\n'.join(texts), 'input_format': 'plain',
            'image_size': list(size), 'seconds': round(seconds, 4),
            'coordinate_space': 'Pixels after automatic document orientation correction',
            'raw': native}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, 'paddleocr-v4')
    parser.add_argument('--dpi', type=int, default=144)
    parser.add_argument('--det-limit', type=int, default=2048,
                        help='Maximum detector image side, in pixels')
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--textline-orientation', action=argparse.BooleanOptionalAction,
                        default=False, help='Enable per-line 180-degree orientation correction (default: disabled)')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--setup-only', action='store_true')
    args = parser.parse_args()
    if args.dpi < 72 or args.det_limit < 32 or args.cpu_threads < 1:
        parser.error('Require dpi >= 72, det-limit >= 32, cpu-threads >= 1')
    base, output, files = discover(args, parser)
    if args.dry_run or (not files and not args.smoke_test and not args.setup_only):
        return 0
    if PACKAGES.is_dir():
        sys.path.insert(0, str(PACKAGES))
    model_cache = ROOT / '.cache/paddlex'
    os.environ['PADDLE_PDX_CACHE_HOME'] = str(model_cache)
    os.environ.setdefault('PADDLE_PDX_MODEL_SOURCE', 'BOS')
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    import numpy as np
    from importlib.metadata import distributions, version
    from paddleocr import PaddleOCR

    options = dict(
        lang='en', ocr_version='PP-OCRv4', device='cpu', engine='paddle_static',
        enable_mkldnn=False, cpu_threads=args.cpu_threads,
        use_doc_orientation_classify=True, use_doc_unwarping=False,
        use_textline_orientation=args.textline_orientation,
        doc_orientation_classify_model_name=MODELS[2],
        textline_orientation_model_name=MODELS[3],
        text_det_limit_side_len=args.det_limit, text_det_limit_type='max',
        text_det_thresh=0.3, text_det_box_thresh=0.6, text_det_unclip_ratio=1.5,
        text_rec_score_thresh=0.5, text_recognition_batch_size=6,
        textline_orientation_batch_size=6,
    )
    started = time.monotonic()
    print('Loading English PP-OCRv4 on CPU', flush=True)
    engine = PaddleOCR(**options)
    if (engine._params['text_detection_model_name'], engine._params['text_recognition_model_name']) != MODELS[:2]:
        raise ValueError('Unexpected OCR model selection; refusing to label it PP-OCRv4')
    active_models = MODELS[:3] + (MODELS[3:] if args.textline_orientation else ())
    model_hashes = {}
    for name in active_models:
        folder = model_cache / 'official_models' / name
        artifacts = [p for p in folder.rglob('*') if p.is_file() and p.suffix in {'.json', '.pdiparams', '.pdmodel', '.yml', '.yaml'}]
        if not artifacts or not any(p.suffix == '.pdiparams' for p in artifacts):
            raise ValueError(f'Model weights missing from expected cache: {folder}')
        model_hashes[name] = {str(p.relative_to(folder)): sha(p) for p in sorted(artifacts)}
    config = {'backend': 'paddleocr-v4', 'runtime': 'paddle_static',
              'runtime_versions': {p: version(p) for p in ('paddleocr', 'paddlepaddle', 'paddlex', 'numpy', 'pypdfium2', 'Pillow')},
              'model': 'PP-OCRv4', 'models': list(active_models), 'model_sha256': model_hashes,
              'options': options, 'dpi': args.dpi, 'schema_version': 1,
              'text_serialization': 'Native recognition order, one detection per line; plain text'}
    output.mkdir(parents=True, exist_ok=True)
    engine.export_paddlex_config_to_yaml(str(output / 'pipeline.yaml'))
    write_json(output / 'environment.json', {
        'python': sys.version, 'platform': platform.platform(),
        'packages': {d.metadata['Name']: d.version for d in distributions(path=[str(PACKAGES)])},
        'model_load_seconds': round(time.monotonic() - started, 4),
    })
    if args.setup_only:
        print('Models loaded and hashed successfully.', flush=True)
        return 0
    if args.smoke_test:
        from PIL import Image, ImageDraw
        image = Image.new('RGB', (900, 180), 'white')
        ImageDraw.Draw(image).text((40, 50), 'Paddle OCR test 12345', fill='black', font_size=44)
        started = time.monotonic()
        result, = engine.predict(np.asarray(image)[:, :, ::-1].copy())
        response = response_from_result(result, 1, image.size, time.monotonic() - started)
        write_json(output / 'smoke-test.json', response)
        if '12345' not in response['text']:
            raise RuntimeError(f'Smoke test failed: {response["text"]!r}')
        print(f'Smoke test passed: {response["text"]}', flush=True)
        return 0

    records, failures = [], []
    for number, path in enumerate(files, 1):
        print(f'[{number}/{len(files)}] {path.relative_to(base)}', flush=True)
        page_number = 0

        def units():
            nonlocal page_number
            for page_number, page in enumerate(pages(path, args.dpi), 1):
                yield page

        def extract(page):
            started = time.monotonic()
            # PaddleX's ndarray input uses OpenCV's BGR channel convention.
            result, = engine.predict(np.asarray(page)[:, :, ::-1].copy())
            response = response_from_result(result, page_number, page.size, time.monotonic() - started)
            print(f'  Page {page_number}: {len(response["raw"]["rec_texts"])} lines, {response["seconds"]:.2f}s', flush=True)
            return response

        try:
            destination = process_file(path, base, output, config, units(), extract,
                                       lambda r: r['text'], args.overwrite)
            paths = sorted(destination.glob('response-*.json'))
            if len(paths) != page_number:
                raise ValueError('Checkpoint count differs from source page count')
            # document.md is the shared runner's convenience export. This .txt
            # and response text are authoritative for plain-text evaluation.
            responses = [json.loads(p.read_text()) for p in paths]
            if [r['page'] for r in responses] != list(range(1, page_number + 1)):
                raise ValueError('Checkpoint page order mismatch')
            (destination / 'document.txt').write_text('\n\n'.join(r['text'] for r in responses), encoding='utf-8')
            artifacts = [destination / 'source.json', *paths]
            records.append({'source': str(path.relative_to(base)), 'source_sha256': sha(path),
                            'run': destination.relative_to(output).as_posix(), 'pages': page_number,
                            'inference_seconds': round(sum(r['seconds'] for r in responses), 4),
                            'artifact_sha256': {p.relative_to(output).as_posix(): sha(p) for p in artifacts}})
            print(f'  Saved: {destination}', flush=True)
        except Exception as error:
            failures.append({'source': str(path.relative_to(base)), 'error': f'{type(error).__name__}: {error}'})
            print(f'  FAILED: {failures[-1]["error"]}', flush=True)
    write_json(output / 'manifest.json', {'backend': 'paddleocr-v4', 'documents': records, 'failures': failures})
    print(f'Completed: {len(records)}; failed: {len(failures)}', flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
