#!/usr/bin/env python3
"""Run DeepSeek OCR 2 locally on Apple Silicon, checkpointing each PDF page."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / '.cache' / 'deepseek-ocr2-packages'
if PACKAGES.is_dir():
    sys.path.insert(0, str(PACKAGES))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from ocr_common import add_arguments, discover, pages, process_file, write_json

MODEL = 'mlx-community/DeepSeek-OCR-2-bf16'
REVISION = '9946f9ac306378a3e6a86cad7d7f8be8e536f092'
PROMPT = '<|grounding|>Convert the document to markdown.'


def clean_markdown(text):
    text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text, flags=re.S)
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text, flags=re.S)
    return text.replace('<|endoftext|>', '').strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, 'deepseek-ocr2')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--dpi', type=int, default=144)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--prompt', default=PROMPT)
    parser.add_argument('--no-cropping', action='store_true', help='Use only the full-page view')
    parser.add_argument('--setup-only', action='store_true', help='Download and load model without processing documents')
    args = parser.parse_args()
    if args.dpi < 72 or args.max_tokens < 1:
        parser.error('--dpi must be at least 72 and --max-tokens positive')
    base, output, files = discover(args, parser)
    if args.dry_run or (not files and not args.setup_only):
        return 0
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from importlib.metadata import version

    if not mx.metal.is_available():
        parser.error('Metal GPU unavailable. Run outside the restricted sandbox on an Apple Silicon Mac.')
    print(f'Loading {args.model}', flush=True)
    model_path = snapshot_download(args.model, revision=args.revision,
                                   allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
    model, processor = load(model_path)
    if args.setup_only:
        print(f'Model loaded successfully: {model_path}', flush=True)
        return 0
    prompt = apply_chat_template(processor, model.config, args.prompt, num_images=1)
    config = {'backend': 'deepseek-ocr2', 'runtime': 'mlx-vlm',
              'runtime_version': version('mlx-vlm'), 'model': args.model,
              'revision': Path(model_path).name, 'dpi': args.dpi,
              'max_tokens': args.max_tokens, 'temperature': 0.0, 'prompt': args.prompt}
    if args.repetition_penalty != 1.0:
        config['repetition_penalty'] = args.repetition_penalty
    if args.no_cropping:
        config['cropping'] = False
    failed = 0
    for index, path in enumerate(files, 1):
        print(f'[{index}/{len(files)}] {path.name}', flush=True)
        page_number = 0

        def units():
            nonlocal page_number
            for page_number, page in enumerate(pages(path, args.dpi), 1):
                yield page

        def extract(page):
            started = time.monotonic()
            result = generate(model, processor, prompt=prompt, image=page,
                              max_tokens=args.max_tokens, temperature=0.0, verbose=False,
                              repetition_penalty=args.repetition_penalty,
                              cropping=not args.no_cropping)
            text = result.text
            tokens = int(result.generation_tokens)
            response = {'page': page_number, 'text': text, 'markdown': clean_markdown(text),
                        'generation_tokens': tokens, 'seconds': round(time.monotonic() - started, 2),
                        'truncated': tokens >= args.max_tokens}
            if response['truncated']:
                print(f'  WARNING: Page {page_number} reached token limit; preserved for review.', flush=True)
            print(f'  Page {page_number}: {tokens} tokens, {response["seconds"]}s', flush=True)
            mx.clear_cache()
            return response

        try:
            destination = process_file(path, base, output, config, units(), extract,
                                       lambda r: ('> WARNING: OCR reached the token limit; this page is incomplete.\n\n'
                                                  if r.get('truncated') else '') + r['markdown'], args.overwrite)
            capped = [json.loads(p.read_text())['page'] for p in sorted(destination.glob('response-*.json'))
                      if json.loads(p.read_text()).get('truncated')]
            source = json.loads((destination / 'source.json').read_text())
            source.update(status='needs_review' if capped else 'complete', truncated_pages=capped)
            write_json(destination / 'source.json', source)
            if capped:
                failed += 1
                print(f'  Needs review: token limit reached on pages {capped}', flush=True)
            print(f'  Saved: {destination}', flush=True)
        except Exception as error:
            failed += 1
            print(f'  FAILED: {type(error).__name__}: {error}', flush=True)
    print(f'Completed without token-limit errors: {len(files) - failed}; failed or needs review: {failed}.', flush=True)
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
