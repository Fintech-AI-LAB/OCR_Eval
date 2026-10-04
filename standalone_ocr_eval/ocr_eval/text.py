import collections
import difflib
import unicodedata
import re
from html.parser import HTMLParser
from .ontology import DEFAULT_ONTOLOGY, DEFAULT_ENTITY_WEIGHT, ontology_metrics
from .field_values import extract_field_values, field_value_metrics, validate_ontology_mode
from .regions import extract_ontology_regions, region_metrics, validate_region_context, DEFAULT_REGION_CONTEXT_TOKENS

# Parse only known formatting tags; preserve OCR placeholders such as <REDACTED>.
FORMATTING_TAGS = set("p div span br hr table thead tbody tfoot tr td th caption h1 h2 h3 h4 h5 h6 b strong i em u s del sup sub ul ol li pre code blockquote a img html body section article".split())
INLINE_TAGS = set('span b strong i em u s del sup sub code a'.split())
ALGORITHM_VERSION = '6.1'

class Plain(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []
    def handle_data(self, data):
        self.parts.append(data)
    def handle_starttag(self, tag, attrs):
        self.parts.append('' if tag in INLINE_TAGS else ' ' if tag in FORMATTING_TAGS else self.get_starttag_text())
    def handle_endtag(self, tag):
        self.parts.append('' if tag in INLINE_TAGS else ' ' if tag in FORMATTING_TAGS else f'</{tag}>')


def extract_text(text, input_format='markdown'):
    """Decode a source format ONCE. Plain text is never parsed as markup."""
    if input_format not in ('plain', 'markdown', 'html'):
        raise ValueError(f'Unsupported input_format: {input_format}')
    if input_format == 'plain':
        return text
    if input_format == 'markdown':
        text = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
        text = re.sub(r'(?m)^\s{0,3}#{1,6}\s+', '', text)
        text = re.sub(r'(?m)^\s*[-*+]\s+(?=\D)', '', text)
        text = re.sub(r'(?m)^\s*\|?[ :|-]+\|[ :|-]*$', '', text)
        text = text.replace('|', ' ').replace('**', '').replace('`', '')
    parser = Plain(); parser.feed(text); parser.close()
    return ''.join(parser.parts)


def clean(text):
    """Idempotent plain-text normalization; does not interpret HTML or Markdown."""
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).replace('\u00ad', '')).strip()


def normalize_texts(texts, formats=None):
    """Extract once, then repair line hyphens only with cross-output corroboration."""
    formats = formats if formats is not None else ['markdown'] * len(texts)
    if len(formats) != len(texts):
        raise ValueError('One input format is required per text.')
    extracted = [extract_text(t, f) for t, f in zip(texts, formats)]
    vocabularies = [set(re.findall(r'[^\W\d_]+', t.casefold())) for t in extracted]
    normalized = []
    for index, text in enumerate(extracted):
        peers = set().union(*(v for i, v in enumerate(vocabularies) if i != index))
        def repair(match):
            joined = match[1] + match[2]
            return joined if joined.casefold() in peers else match[0]
        text = re.sub(r'([^\W\d_]+)-[ \t]*\r?\n[ \t]*([^\W\d_]+)', repair, text)
        normalized.append(clean(text))
    return normalized


# Keep signed numbers, internal decimal/date/ID punctuation, and percentages intact.
SIGNED_NUMBER = r'[+\-−]?\d+(?:[.,:/-]\d+)*(?:[%‰])?'
# Preserve accounting parentheses literally; do not infer that every (123) is negative.
NUMBER = r'(?:\(\s*' + SIGNED_NUMBER + r'\s*\)|' + SIGNED_NUMBER + r')'
TOKEN = re.compile(r'(?<!\w)' + NUMBER + r'(?!\w)|\w+')


def tokens(text):
    return [re.sub(r'\s+', '', t) if t.startswith('(') else t
            for t in TOKEN.findall(text.casefold().replace('−', '-'))]


def numeric_strings(text):
    return sorted(numeric_counts(text))


def numeric_counts(text):
    return collections.Counter(re.sub(r'\s+', '', n) for n in
                               re.findall(r'(?<!\w)' + NUMBER + r'(?!\w)', text.replace('−', '-')))


def counter_overlap(left, right):
    total = sum(left.values()) + sum(right.values())
    return 2 * sum((left & right).values()) / total if total else 0.0

def compare_page(doc, index, a, b, left, right, ontology_path=DEFAULT_ONTOLOGY,
                 entity_weight=DEFAULT_ENTITY_WEIGHT, *, left_fields=None, right_fields=None,
                 left_regions=None, right_regions=None, ontology_mode='regions',
                 region_context_tokens=DEFAULT_REGION_CONTEXT_TOKENS):
    validate_ontology_mode(ontology_mode)
    validate_region_context(region_context_tokens)
    x, y = tokens(left), tokens(right)
    cx, cy = collections.Counter(x), collections.Counter(y)
    overlap = sum((cx & cy).values())
    denom = len(x) + len(y)
    # Symmetrize SequenceMatcher because its tie-breaking may depend on argument order.
    forward = difflib.SequenceMatcher(None, x, y, autojunk=False)
    reverse = difflib.SequenceMatcher(None, y, x, autojunk=False)
    changes = []
    for tag, i, j, k, l in forward.get_opcodes():
        if tag != 'equal':
            changes.append({'kind': tag, 'left_tokens': x[i:j], 'right_tokens': y[k:l],
                            'left_offset': i, 'right_offset': k,
                            'left_context': ' '.join(x[max(0,i-5):min(len(x),j+5)]),
                            'right_context': ' '.join(y[max(0,k-5):min(len(y),l+5)])})
    nx, ny = numeric_counts(left), numeric_counts(right)
    empty_fields = {'fields': [], 'issues': []}
    if ontology_mode != 'values':
        left_fields = right_fields = empty_fields
    fields = field_value_metrics(
        left_fields if left_fields is not None else extract_field_values(left, ontology_path),
        right_fields if right_fields is not None else extract_field_values(right, ontology_path))
    regions = region_metrics(
        left_regions if left_regions is not None else extract_ontology_regions(left, ontology_path if ontology_mode == 'regions' else None, context_tokens=region_context_tokens),
        right_regions if right_regions is not None else extract_ontology_regions(right, ontology_path if ontology_mode == 'regions' else None, context_tokens=region_context_tokens))
    return {**ontology_metrics(x, y, ontology_path, entity_weight), **fields, **regions,
            'algorithm_version': ALGORITHM_VERSION, 'ontology_mode': ontology_mode,
            'document':doc, 'page':index, 'left':a, 'right':b,
            'left_tokens':len(x), 'right_tokens':len(y), 'shared_token_occurrences':overlap,
            'token_overlap':2*overlap/denom if denom else 0,
            'sequence_similarity':(forward.ratio()+reverse.ratio())/2 if denom else 0,
            'numeric_agreement':counter_overlap(nx, ny) if nx or ny else None,
            'only_left_numeric_strings':sorted(nx-ny), 'only_right_numeric_strings':sorted(ny-nx),
            'unmatched_left_numeric_counts':dict(nx-ny), 'unmatched_right_numeric_counts':dict(ny-nx),
            'changes':changes}
