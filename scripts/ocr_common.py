"""Shared discovery, page decoding, and result storage for both OCR backends."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'}


def add_arguments(parser, backend):
    parser.add_argument('--input', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / backend)
    parser.add_argument('--limit', type=int, help='Maximum number of source files')
    parser.add_argument('--dry-run', action='store_true', help='List files without loading models or calling APIs')
    parser.add_argument('--overwrite', action='store_true', help='Recompute cached responses')


def discover(args, parser):
    source, output = args.input.resolve(), args.output.resolve()
    if not source.exists():
        parser.error(f'Input does not exist: {source}')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    candidates = [source] if source.is_file() else source.rglob('*')
    files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in EXTENSIONS
                   and not p.is_relative_to(output))
    if args.limit:
        files = files[:args.limit]
    base = source.parent if source.is_file() else source
    print(f'Found {len(files)} supported files.', flush=True)
    if args.dry_run:
        for path in files:
            print(path.relative_to(base))
    return base, output, files


def image_pages(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        for index in range(getattr(image, 'n_frames', 1)):
            image.seek(index)
            yield ImageOps.exif_transpose(image).convert('RGB')


def pages(path, dpi=144):
    if path.suffix.lower() != '.pdf':
        yield from image_pages(path)
        return
    import pypdfium2 as pdfium
    with pdfium.PdfDocument(path) as document:
        for index in range(len(document)):
            page = document[index]
            try:
                bitmap = page.render(scale=dpi / 72)
                try:
                    image = bitmap.to_pil().convert('RGB').copy()
                finally:
                    bitmap.close()
            finally:
                page.close()
            yield image


def destination_for(path, base, output, config):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    digest.update(json.dumps(config, sort_keys=True).encode())
    destination = output / path.relative_to(base) / digest.hexdigest()[:16]
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def process_file(path, base, output, config, units, extract, markdown, overwrite=False):
    destination = destination_for(path, base, output, config)
    count = 0
    # Stream Markdown instead of retaining every page's image base64 in RAM.
    temporary = destination / 'document.md.tmp'
    with temporary.open('w', encoding='utf-8') as document:
        for count, unit in enumerate(units, 1):
            checkpoint = destination / f'response-{count:04d}.json'
            if checkpoint.exists() and not overwrite:
                response = json.loads(checkpoint.read_text(encoding='utf-8'))
            else:
                response = extract(unit)
                write_json(checkpoint, response)
            if count > 1:
                document.write('\n\n---\n\n')
            document.write(markdown(response))
    temporary.replace(destination / 'document.md')
    write_json(destination / 'source.json', {
        'source': str(path.relative_to(base)), **config, 'responses': count,
    })
    return destination
