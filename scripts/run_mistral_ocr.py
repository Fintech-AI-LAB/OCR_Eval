#!/usr/bin/env python3
"""OCR PDFs and images recursively with Mistral; checkpoint every API response."""
import argparse
import base64
import io
import os

from ocr_common import add_arguments, discover, image_pages, process_file


def documents(path):
    if path.suffix.lower() == '.pdf':
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        yield {'type': 'document_url', 'document_url': f'data:application/pdf;base64,{data}'}
        return
    for page in image_pages(path):
        with io.BytesIO() as buffer:
            page.save(buffer, format='PNG')
            data = base64.b64encode(buffer.getvalue()).decode('ascii')
        yield {'type': 'image_url', 'image_url': f'data:image/png;base64,{data}'}


def markdown(response):
    pages = []
    for page in response.get('pages', []):
        text = page.get('markdown', '')
        for table in page.get('tables') or []:
            table_id = table.get('id')
            if table_id:
                text = text.replace(f'[{table_id}]({table_id})', table.get('content', ''))
        for image in page.get('images') or []:
            image_id, data = image.get('id'), image.get('image_base64')
            if image_id and data:
                if not data.startswith('data:'):
                    import mimetypes
                    mime = mimetypes.guess_type(image_id)[0] or 'image/jpeg'
                    data = f'data:{mime};base64,{data}'
                text = text.replace(f']({image_id})', f']({data})')
        pages.append(text)
    return '\n\n---\n\n'.join(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, 'mistral')
    parser.add_argument('--model', default='mistral-ocr-latest')
    args = parser.parse_args()
    source, output, files = discover(args, parser)
    if args.dry_run or not files:
        return 0
    api_key = os.environ.get('MISTRAL_API_KEY')
    if not api_key:
        parser.error('Set the MISTRAL_API_KEY environment variable before running.')
    try:
        from mistralai.client import Mistral
    except ImportError:
        parser.error('Install requirements-mistral.txt in Conda environment myenv3.13 first.')
    failed = 0
    with Mistral(api_key=api_key) as client:
        for number, path in enumerate(files, 1):
            relative = path.relative_to(source)
            print(f'[{number}/{len(files)}] {relative}', flush=True)
            try:
                def extract(document):
                    return client.ocr.process(
                        model=args.model, document=document,
                        table_format='html', include_image_base64=True,
                    ).model_dump(mode='json')
                destination = process_file(
                    path, source, output,
                    {'backend': 'mistral', 'model': args.model,
                     'table_format': 'html', 'include_image_base64': True},
                    documents(path), extract, markdown, args.overwrite,
                )
                print(f'  Saved: {destination}', flush=True)
            except Exception as error:
                # Avoid logging response bodies, which may contain document data.
                failed += 1
                status = getattr(error, 'status_code', None)
                print(f'  FAILED: {type(error).__name__}' + (f' (HTTP {status})' if status else ''), flush=True)
    print(f'Completed: {len(files) - failed}; failed: {failed}.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
