"""Page-image Claude OCR through the user-selected OpenAI-compatible gateway."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import version
import io
import json
import os
import re
from pathlib import Path
import time

from ocr_common import ROOT, destination_for, write_json
from run_astra_ocr import DEFAULT_FILES, PROMPT, SCHEMA, rendered_pages

MODEL = 'claude-fable-5-1'
BASE_URL = 'https://one.liangzhili.com/v1'


def safe_error_message(error):
    body = getattr(error, 'body', None)
    if not isinstance(body, dict):
        return None
    detail = body.get('error', body)
    message = detail.get('message') if isinstance(detail, dict) else None
    if not isinstance(message, str):
        return None
    key = os.environ.get('OPENAI_API_KEY')
    if key:
        message = message.replace(key, '[redacted]')
    return re.sub(r'sk-[A-Za-z0-9_*.-]+', '[redacted]', message)[:500]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_response(raw, model):
    if raw.get('model') != model or len(raw.get('choices', [])) != 1:
        raise ValueError('Unexpected returned model or choice count')
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop' or choice['message'].get('refusal'):
        raise ValueError('Incomplete or refused response')
    text = choice['message']['content']
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or set(parsed) != {'blocks', 'uncertainties'}:
        raise ValueError('Unexpected structured output keys')
    if not isinstance(parsed['blocks'], list) or not isinstance(parsed['uncertainties'], list):
        raise ValueError('Structured output must contain arrays')
    allowed = SCHEMA['properties']['blocks']['items']['properties']['type']['enum']
    for block in parsed['blocks']:
        if (not isinstance(block, dict) or set(block) != {'type', 'markdown'} or
                block['type'] not in allowed or not isinstance(block['markdown'], str)):
            raise ValueError('Invalid transcription block')
    if not all(isinstance(x, str) for x in parsed['uncertainties']):
        raise ValueError('Invalid uncertainty notes')
    return parsed


def validate_checkpoint(record, page, model):
    if record.get('page_number') != page:
        raise ValueError('Checkpoint page mismatch')
    parsed = parse_response(record['raw_response'], model)
    if any(record[k] != parsed[k] for k in ('blocks', 'uncertainties')):
        raise ValueError('Checkpoint differs from raw provider response')
    if record['markdown'] != '\n\n'.join(b['markdown'] for b in parsed['blocks']):
        raise ValueError('Markdown differs from provider blocks')


def infer(client, image, page, config):
    with io.BytesIO() as buffer:
        image.save(buffer, format='PNG')
        png = buffer.getvalue()
    text = f'Transcribe physical page {page}.'
    if config['json_mode'] == 'json_object':
        text += '\nReturn a JSON object conforming to this schema: ' + json.dumps(SCHEMA)
        response_format = {'type': 'json_object'}
    else:
        response_format = {'type': 'json_schema', 'json_schema': {
            'name': 'ocr_page', 'strict': True, 'schema': SCHEMA}}
    started = time.monotonic()
    result = client.chat.completions.create(
        model=config['model'], max_tokens=config['max_output_tokens'],
        messages=[{'role': 'system', 'content': PROMPT},
                  {'role': 'user', 'content': [
                      {'type': 'text', 'text': text},
                      {'type': 'image_url', 'image_url': {
                          'url': 'data:image/png;base64,' + base64.b64encode(png).decode('ascii'),
                          'detail': 'high'}},
                  ]}],
        response_format=response_format,
    )
    return {'raw_response': result.model_dump(mode='json'), 'page_number': page,
            'seconds': round(time.monotonic() - started, 4), 'image_size': list(image.size),
            'image_sha256': hashlib.sha256(png).hexdigest(),
            'request_id': getattr(result, '_request_id', None)}


def process_document(path, args, client, config):
    path = path.resolve()
    folder = destination_for(path, path.parent, args.output.resolve(), config)
    responses = []
    failed_pages = []
    try:
        for page, image in enumerate(rendered_pages(path, args.dpi), 1):
            checkpoint = folder / f'response-{page:04d}.json'
            failure_path = folder / f'failed-{page:04d}.json'
            if failure_path.exists() and not args.overwrite:
                failed_pages.append(json.loads(failure_path.read_text()))
                print(f'{path.name}: retained failure for page {page}', flush=True)
                continue
            if checkpoint.exists() and not args.overwrite:
                record = json.loads(checkpoint.read_text())
            else:
                print(f'{path.name}: requesting page {page}', flush=True)
                try:
                    record = infer(client, image, page, config)
                except Exception as error:
                    if getattr(error, 'status_code', None) != 400:
                        raise
                    failure = {'page_number': page, 'error_type': type(error).__name__,
                               'http_status': 400, 'message': safe_error_message(error)}
                    write_json(failure_path, failure)
                    failed_pages.append(failure)
                    print(f'{path.name}: failed page {page}: ' + json.dumps(failure), flush=True)
                    continue
                try:
                    parsed = parse_response(record['raw_response'], config['model'])
                    record.update(parsed, markdown='\n\n'.join(b['markdown'] for b in parsed['blocks']))
                    validate_checkpoint(record, page, config['model'])
                except (ValueError, KeyError, TypeError):
                    write_json(folder / f'incomplete-{page:04d}.json', record)
                    raise
                write_json(checkpoint, record)
            validate_checkpoint(record, page, config['model'])
            responses.append(record)
            print(f'{path.name}: saved page {page}, {record["seconds"]:.1f}s', flush=True)
        if failed_pages:
            return {'source': path.name, 'error_type': 'IncompleteDocument',
                    'run': str(folder.relative_to(args.output.resolve())),
                    'completed_pages': len(responses), 'failed_pages': failed_pages}
        paths = sorted(folder.glob('response-*.json'))
        if len(paths) != len(responses):
            raise ValueError('Unexpected extra page checkpoints')
        source_hash = sha(path)
        write_json(folder / 'source.json', {'source': path.name, 'source_sha256': source_hash,
                   **config, 'responses': len(responses), 'artifact_sha256': {p.name: sha(p) for p in paths}})
        write_json(folder / 'document.json', {'source': path.name, 'source_sha256': source_hash,
                   **config, 'pages': [{k: v for k, v in r.items() if k != 'raw_response'} for r in responses]})
        (folder / 'document.md').write_text('\n\n'.join(
            f'<!-- page:{r["page_number"]} -->\n\n{r["markdown"]}' for r in responses), encoding='utf-8')
        return {'source': path.name, 'source_sha256': source_hash, 'pages': len(responses),
                'run': str(folder.relative_to(args.output.resolve())), 'source_metadata_sha256': sha(folder / 'source.json')}
    except Exception as error:
        failure = {'source': path.name, 'error_type': type(error).__name__,
                   'http_status': getattr(error, 'status_code', None), 'code': getattr(error, 'code', None),
                   'message': safe_error_message(error)}
        print('FAILED: ' + json.dumps(failure), flush=True)
        return failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL', BASE_URL))
    parser.add_argument('--input', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/claude-vlm-fable-5-1')
    parser.add_argument('--dpi', type=int, default=144)
    parser.add_argument('--workers', type=int, choices=[1, 2, 3], default=3)
    parser.add_argument('--json-mode', choices=['json_schema', 'json_object'], default='json_schema')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    files = args.input or [ROOT / 'data' / name for name in DEFAULT_FILES]
    if args.dpi < 72:
        parser.error('dpi must be at least 72')
    if len({p.resolve() for p in files}) != len(files):
        parser.error('Duplicate inputs')
    for path in files:
        if not path.is_file() or path.suffix.lower() != '.pdf':
            parser.error(f'Expected PDF: {path}')
    if args.dry_run:
        for path in files:
            print(path)
        return 0
    if not os.environ.get('OPENAI_API_KEY'):
        parser.error('Set OPENAI_API_KEY locally')
    from openai import OpenAI
    config = {'backend': 'claude-vlm-gateway', 'model': args.model, 'api_base': args.base_url.rstrip('/'),
              'api': 'chat.completions', 'sdk_version': version('openai'), 'dpi': args.dpi,
              'schema_version': 1, 'prompt': PROMPT, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
              'response_schema': SCHEMA, 'json_mode': args.json_mode, 'image_detail': 'high',
              'max_output_tokens': 16000, 'reasoning': 'provider default; no reasoning parameter sent',
              'provenance_limit': 'Model identity is reported by the gateway; upstream inference is not independently verified.'}
    args.output.mkdir(parents=True, exist_ok=True)
    with OpenAI(base_url=args.base_url, timeout=300, max_retries=2) as client:
        if args.smoke_test:
            from PIL import Image, ImageDraw
            image = Image.new('RGB', (900, 180), 'white')
            ImageDraw.Draw(image).text((40, 50), 'Claude OCR test 12345', fill='black', font_size=44)
            record = infer(client, image, 1, config)
            write_json(args.output / 'smoke-test.json', record)
            parsed = parse_response(record['raw_response'], args.model)
            if '12345' not in '\n'.join(b['markdown'] for b in parsed['blocks']):
                raise ValueError('Synthetic image transcription failed')
            print(f'Smoke test passed: returned model {record["raw_response"]["model"]}', flush=True)
            return 0
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(lambda path: process_document(path, args, client, config), files))
    failures = [r for r in records if 'error_type' in r]
    write_json(args.output / 'manifest.json', {'model': args.model, 'api_base': config['api_base'],
               'documents': [r for r in records if 'error_type' not in r], 'failures': failures,
               'workers': args.workers, 'wall_seconds': round(time.monotonic() - started, 4)})
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
