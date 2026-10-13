"""Pinned BF16 Unlimited-OCR on Apple Silicon, with validated page checkpoints."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import time

from ocr_common import ROOT, add_arguments, discover, pages, destination_for, write_json

for packages in [ROOT / '.cache/deepseek-ocr2-packages', ROOT / '.cache/unlimited-ocr-packages']:
    if packages.is_dir():
        sys.path.insert(0, str(packages))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
MODEL = 'mlx-community/Unlimited-OCR-bf16'
REVISION = '6d9f675e3fa73dd49cd03f630868b1941c72803f'
UPSTREAM_REVISION = '07dea832e22aefee32ad281d4b80551282e1c168'
METHOD = 'unlimited-ocr-bf16'
PROMPT = 'document parsing.'
DET = re.compile(r'<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_orientation(manifest_path, source_paths, dpi):
    """Replay only verified PP-LCNet page angles, never another engine's text."""
    from evaluate_paddleocr_v4 import verify_run
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest['failures']:
        raise ValueError('Orientation manifest has failed documents')
    documents, model_hashes = {}, None
    for name, source in source_paths.items():
        records = [r for r in manifest['documents'] if r['source'] == name]
        if len(records) != 1:
            raise ValueError(f'Require exactly one orientation run for {name}')
        record = records[0]
        responses, metadata = verify_run(record, source, manifest_path.parent, record['pages'])
        if metadata['dpi'] != dpi or not metadata['options']['use_doc_orientation_classify']:
            raise ValueError('Orientation DPI/options mismatch')
        classifier = 'PP-LCNet_x1_0_doc_ori'
        if metadata['options']['doc_orientation_classify_model_name'] != classifier:
            raise ValueError('Unexpected orientation classifier')
        hashes = metadata['model_sha256'][classifier]
        if model_hashes is not None and model_hashes != hashes:
            raise ValueError('Orientation classifier differs between documents')
        model_hashes = hashes
        angles = [r['raw']['doc_preprocessor_res']['angle'] for r in responses]
        if any(type(a) is not int or a not in (0, 90, 180, 270) for a in angles):
            raise ValueError('Unsupported orientation angle')
        documents[name] = {'source_sha256': sha(source), 'angles_ccw': angles,
                           'input_sizes': [r['image_size'] for r in responses]}
    return {'method': 'verified_saved_document_orientation', 'classifier': classifier,
            'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
            'classifier_files_sha256': model_hashes, 'documents': documents,
            'transform': 'lossless PIL transpose; angles counterclockwise; no deskew/unwarping'}


def orient_page(image, source_name, page, orientation):
    if orientation is None:
        return image, 0
    document = orientation['documents'][source_name]
    if list(image.size) != document['input_sizes'][page - 1]:
        raise ValueError('Orientation input dimensions mismatch')
    angle = document['angles_ccw'][page - 1]
    from PIL import Image
    transforms = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
                  270: Image.Transpose.ROTATE_270}
    return (image.transpose(transforms[angle]) if angle else image), angle


def clean_markdown(raw):
    """Upstream block cleanup; drop image blocks, retain text without repairs."""
    blocks, current = [], None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        match = DET.match(line)
        if match:
            if current is not None:
                blocks.append(current)
            category, content = match.groups()
            current = None if category == 'image' else ([content.strip()] if content.strip() else [])
            continue
        if current is None:
            current = []
        current.append(line)
    if current is not None:
        blocks.append(current)
    return '\n\n'.join('\n'.join(block) for block in blocks).strip()


def banned_tokens(sequence, size=35, window=128):
    """Same sliding window boundaries as Baidu's inference processor."""
    if len(sequence) < size:
        return set()
    start, end = max(0, len(sequence) - window), len(sequence) - size + 1
    prefix = tuple(sequence[-(size - 1):]) if size > 1 else ()
    return {sequence[i + size - 1] for i in range(start, end)
            if size == 1 or tuple(sequence[i:i + size - 1]) == prefix}


class NoRepeatNgram:
    def __call__(self, tokens, logits):
        banned = banned_tokens(tokens.tolist())
        if banned:
            import mlx.core as mx
            mask = mx.any(mx.arange(logits.shape[-1])[:, None] == mx.array(sorted(banned))[None, :], axis=1)
            logits = mx.where(mask, -float('inf'), logits)
        return logits


def validate_response(response, page, image_hash=None):
    if response['page'] != page:
        raise ValueError('Page order mismatch')
    if image_hash is not None and response['image_sha256'] != image_hash:
        raise ValueError('Rendered image hash mismatch')
    if response['markdown'] != clean_markdown(response['text']):
        raise ValueError('Cleaned text differs from native model text')
    if response['finish_reason'] != 'stop' or response['truncated']:
        raise ValueError('Incomplete generation; rerun with --overwrite or a larger budget')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, METHOD)
    parser.add_argument('--dpi', type=int, default=144)
    parser.add_argument('--max-tokens', type=int, default=32768)
    parser.add_argument('--setup-only', action='store_true')
    parser.add_argument('--orientation-manifest', type=Path,
                        help='Replay verified PP-LCNet page angles from a saved PaddleOCR manifest')
    args = parser.parse_args()
    if args.dpi < 72 or args.max_tokens < 1:
        parser.error('Require DPI >=72 and a positive token budget')
    if args.orientation_manifest and args.output == parser.get_default('output'):
        args.output = ROOT / 'output/unlimited-ocr-bf16-rotated'
    base, output, files = discover(args, parser)
    orientation = (load_orientation(args.orientation_manifest,
        {str(p.relative_to(base)): p for p in files}, args.dpi) if args.orientation_manifest else None)
    if args.dry_run:
        return 0
    import mlx.core as mx
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from huggingface_hub import snapshot_download
    if not mx.metal.is_available():
        parser.error('Metal GPU unavailable; inference requires Apple Silicon GPU access')
    started = time.monotonic()
    model_path = Path(snapshot_download(MODEL, revision=REVISION,
        allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', 'README.md']))
    print(f'Loading {MODEL}@{REVISION}', flush=True)
    model, processor = load(model_path)
    if model.config.model_type != 'unlimited-ocr':
        raise ValueError('Unexpected model architecture')
    prompt = apply_chat_template(processor, model.config, PROMPT, num_images=1)
    if prompt != '<image>' + PROMPT:
        raise ValueError(f'Unexpected Unlimited-OCR prompt template: {prompt!r}')
    config = {'backend': METHOD, 'runtime': 'mlx-vlm', 'model': MODEL, 'revision': REVISION,
        'upstream_model': 'baidu/Unlimited-OCR', 'upstream_revision_observed': UPSTREAM_REVISION,
        'precision': 'BF16 community conversion', 'dpi': args.dpi, 'prompt': prompt,
        'max_tokens': args.max_tokens, 'temperature': 0.0, 'image_mode': 'gundam',
        'base_size': 1024, 'image_size': 640, 'cropping': True,
        'no_repeat_ngram_size': 35, 'ngram_window': 128, 'inference_unit': 'single_page',
        'runtime_versions': {p: version(p) for p in ['mlx-vlm', 'mlx', 'transformers', 'pypdfium2', 'Pillow']},
        'model_files_sha256': {p.name: sha(p) for p in sorted(model_path.iterdir()) if p.is_file()},
        'runner_sha256': sha(Path(__file__))}
    if orientation is not None:
        config['orientation'] = orientation
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'model_provenance.json', config)
    if args.setup_only:
        print('Model loaded; prompt and architecture verified.', flush=True)
        return 0
    manifest = {'method': METHOD, 'documents': [], 'failures': [], 'wall_seconds': 0}
    for path in files:
        folder = destination_for(path, base, output, config)
        count, reused = 0, 0
        try:
            for count, image in enumerate(pages(path, args.dpi), 1):
                original_hash = hashlib.sha256(image.tobytes()).hexdigest()
                image, angle = orient_page(image, str(path.relative_to(base)), count, orientation)
                image_hash = hashlib.sha256(image.tobytes()).hexdigest()
                checkpoint = folder / f'response-{count:04d}.json'
                cached = checkpoint.exists() and not args.overwrite
                if cached:
                    response = json.loads(checkpoint.read_text())
                    validate_response(response, count, image_hash)
                    reused += 1
                else:
                    tick = time.monotonic()
                    result = generate(model, processor, prompt=prompt, image=image,
                        max_tokens=args.max_tokens, temperature=0.0, verbose=False,
                        cropping=True, base_size=1024, image_size=640,
                        logits_processors=[NoRepeatNgram()])
                    response = {'page': count, 'text': result.text,
                        'markdown': clean_markdown(result.text), 'image_sha256': image_hash,
                        'image_size': list(image.size), 'generation_tokens': int(result.generation_tokens),
                        'prompt_tokens': int(result.prompt_tokens), 'finish_reason': result.finish_reason,
                        'truncated': result.finish_reason != 'stop',
                        'seconds': round(time.monotonic() - tick, 3)}
                    if orientation is not None:
                        response.update(original_image_sha256=original_hash, rotation_ccw_degrees=angle)
                    write_json(checkpoint, response)
                    validate_response(response, count, image_hash)
                    mx.clear_cache()
                print(f'{path.name}: page {count}, {response["generation_tokens"]} tokens, '
                      f'{response["seconds"]}s' + (' (cached)' if cached else ''), flush=True)
            responses = sorted(folder.glob('response-*.json'))
            if len(responses) != count:
                raise ValueError('Extra page checkpoints')
            (folder / 'document.md').write_text('\n\n---\n\n'.join(
                json.loads(p.read_text())['markdown'] for p in responses), encoding='utf-8')
            metadata = {'source': str(path.relative_to(base)), 'source_sha256': sha(path),
                        'config': config, 'responses': count, 'status': 'complete'}
            write_json(folder / 'source.json', metadata)
            record = {'source': metadata['source'], 'source_sha256': sha(path),
                'run': str(folder.relative_to(output)), 'pages': count, 'reused_pages': reused,
                'artifact_sha256': {p.name: sha(p) for p in [folder / 'source.json', folder / 'document.md', *responses]}}
            manifest['documents'].append(record)
        except Exception as error:
            manifest['failures'].append({'source': str(path.relative_to(base)), 'page': count,
                                         'error': f'{type(error).__name__}: {error}'})
            print(f'FAILED: {path.name}: {error}', flush=True)
        manifest['wall_seconds'] = round(time.monotonic() - started, 3)
        write_json(output / 'manifest.json', manifest)
    print(f'Completed {len(manifest["documents"])} documents; failures: {len(manifest["failures"])}', flush=True)
    return int(bool(manifest['failures']))


if __name__ == '__main__':
    raise SystemExit(main())
