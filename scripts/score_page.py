"""Score agreement among OCR output files for one physical page (0–100)."""
import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

if __package__:
    from .evaluate_outputs import clean, compare_page, tokens
else:
    from evaluate_outputs import clean, compare_page, tokens


def _json_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ('pages', 'full_page_text'):
            if key in value:
                pages = value[key]
                if not isinstance(pages, list) or len(pages) != 1:
                    raise ValueError('JSON must contain exactly one page; split multi-page outputs first.')
                return _json_text(pages[0])
        if isinstance(value.get('markdown'), str):
            text = value['markdown']
            for table in value.get('tables') or []:
                text = text.replace(f'[{table["id"]}]({table["id"]})', table['content'])
            return text
        if isinstance(value.get('text'), str):
            return value['text']
        if isinstance(value.get('lines'), list):
            return '\n'.join(_json_text(line) for line in value['lines'])
    # MinerU's single-page list of content blocks.
    if isinstance(value, list) and all(isinstance(b, dict) and
            isinstance(b.get('content'), (str, type(None))) and 'content' in b for b in value):
        return '\n'.join(b['content'] or '' for b in value)
    raise ValueError('Unsupported page JSON; provide a text/Markdown file or a supported single-page OCR JSON.')


def _read_page(path):
    path = Path(path)
    if path.suffix.lower() not in ('.txt', '.md', '.markdown', '.html', '.htm', '.json'):
        raise ValueError(f'{path}: expected an OCR text, Markdown, HTML or JSON file.')
    text = path.read_text(encoding='utf-8-sig')
    return clean(_json_text(json.loads(text)) if path.suffix.lower() == '.json' else text)


def score_models(file_paths, overlap_weight=0.5):
    """Return one consensus score per input file, in input order.

    Inputs must be distinct OCR output files for the same physical page.
    At least two outputs are required; with two, their scores are identical.
    Raises ValueError when all outputs have no scorable word/number tokens.
    Empty pairs receive zero if another output contains scorable text.
    """
    if not isinstance(file_paths, (list, tuple)) or len(file_paths) < 2:
        raise ValueError('Provide an array of at least two file paths for one page.')
    if not math.isfinite(overlap_weight) or not 0 <= overlap_weight <= 1:
        raise ValueError('overlap_weight must be finite and between 0 and 1.')
    paths = [Path(p).resolve() for p in file_paths]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate file paths would inflate agreement.')
    texts = [_read_page(p) for p in paths]
    has_tokens = [bool(tokens(text)) for text in texts]
    if not any(has_tokens):
        raise ValueError('All outputs lack scorable text; exclude this page from the average.')
    scores = [[] for _ in paths]
    for a, b in itertools.combinations(range(len(paths)), 2):
        if not has_tokens[a] and not has_tokens[b]:
            similarity = 0.0
        else:
            pair = compare_page('page', 1, str(paths[a]), str(paths[b]), texts[a], texts[b])
            similarity = (overlap_weight * pair['token_overlap'] +
                          (1 - overlap_weight) * pair['sequence_similarity'])
        scores[a].append(100 * similarity)
        scores[b].append(100 * similarity)
    return [statistics.mean(values) for values in scores]


def score_page(file_paths, overlap_weight=0.5):
    """Return a single 0–100 score: mean agreement across all output pairs.

    100 means identical normalized tokens, 0 means no agreement. This measures
    consensus, not accuracy. Average these scalar scores to weight pages equally.
    """
    return statistics.mean(score_models(file_paths, overlap_weight))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--overlap-weight', type=float, default=0.5)
    args = parser.parse_args()
    try:
        print(score_page(args.files, args.overlap_weight))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
