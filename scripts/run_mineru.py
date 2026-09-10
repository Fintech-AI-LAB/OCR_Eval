#!/usr/bin/env python3
"""OCR the data folder using the exact MinerU checkpoint on MPS or CPU."""
import argparse
import os
from pathlib import Path
from ocr_common import ROOT, add_arguments, discover, pages, process_file, write_json

MODEL_ID = 'opendatalab/MinerU2.5-Pro-2604-1.2B'
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
# Must be set before torch is imported; unsupported individual ops may use CPU.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, 'mineru')
    parser.add_argument('--files', type=Path, nargs='+', help='Process only these files, loading the model once')
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    parser.add_argument('--dpi', type=int, default=144, help='PDF rendering resolution')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error('--dpi must be positive')
    if not args.smoke_test:
        if args.files:
            files = [p.resolve() for p in args.files]
            for path in files:
                if not path.is_file():
                    parser.error(f'Missing input: {path}')
            source = Path(os.path.commonpath([str(p.parent) for p in files]))
            output = args.output.resolve()
            if args.limit:
                if args.limit < 1:
                    parser.error('--limit must be positive')
                files = files[:args.limit]
            if args.dry_run:
                for path in files:
                    print(path)
        else:
            source, output, files = discover(args, parser)
        if args.dry_run or not files:
            return 0

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from mineru_vl_utils import MinerUClient

    device = args.device
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable; use --device cpu to explicitly run on CPU.')
    print(f'Loading {MODEL_ID} on {device}', flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float32 if device == 'cpu' else torch.float16,
        attn_implementation='eager',
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    client = MinerUClient(backend='transformers', model=model, processor=processor, batch_size=1)
    def extract(page):
        with torch.inference_mode():
            return client.two_step_extract(page)

    if args.smoke_test:
        page = Image.new('RGB', (800, 400), 'white')
        ImageDraw.Draw(page).text((40, 80), 'MinerU test: Hello world 123', fill='black', font_size=32)
        result = extract(page)
        if not result:
            raise RuntimeError('Smoke test returned no blocks.')
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / 'smoke-test.json', result)
        print(f'Smoke test passed on {device}')
        return 0

    from mineru_vl_utils.post_process import json2md
    failed = 0
    for number, path in enumerate(files, 1):
        print(f'[{number}/{len(files)}] {path.relative_to(source)}', flush=True)
        try:
            destination = process_file(
                path, source, output,
                {'backend': 'mineru', 'model': MODEL_ID, 'device': device, 'dpi': args.dpi},
                pages(path, args.dpi), extract, json2md, args.overwrite,
            )
            print(f'  Saved: {destination}', flush=True)
        except Exception as error:
            failed += 1
            print(f'  FAILED: {type(error).__name__}: {error}', flush=True)
    print(f'Completed: {len(files) - failed}; failed: {failed}.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
