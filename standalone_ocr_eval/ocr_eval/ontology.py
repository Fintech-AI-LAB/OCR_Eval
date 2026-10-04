"""Deterministic lexical ontology weighting; no entity-value inference."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

DEFAULT_ONTOLOGY = None
DEFAULT_ENTITY_WEIGHT = 3.0


def validate_entity_weight(weight):
    if not math.isfinite(weight) or weight < 1:
        raise ValueError('entity_weight must be finite and at least 1.')


@lru_cache(maxsize=8)
def _load(path, mtime_ns, size):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    concepts = data.get('concepts') if isinstance(data, dict) else None
    if not isinstance(concepts, dict) or not concepts:
        raise ValueError('Ontology must contain a nonempty concepts mapping.')
    phrases = set()
    for key, concept in concepts.items():
        if not isinstance(concept, dict):
            raise ValueError('Ontology concept records must be objects.')
        names = [key, concept.get('name', key)]
        for field in ('children', 'fields'):
            values = concept.get(field, [])
            if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                raise ValueError(f'Ontology {field} must be a list of strings.')
            names.extend(values)
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError('Ontology names must be nonempty strings.')
            name = unicodedata.normalize('NFKC', name)
            # Accept both repository CamelCase names and their printed word forms.
            split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
            split = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', split)
            for variant in (name, split):
                phrase = tuple(re.findall(r'[^\W_]+', variant.casefold()))
                if phrase:
                    phrases.add(phrase)
    trie = {}
    for phrase in sorted(phrases):
        node = trie
        for token in phrase:
            node = node.setdefault(token, {})
        node[None] = ' '.join(phrase)
    return trie, {'path': path, 'sha256': hashlib.sha256(raw).hexdigest(),
                  'concept_count': len(concepts), 'phrase_count': len(phrases),
                  'matching': 'case-insensitive longest nonoverlapping concept/field phrases; no value inference'}


def load_ontology(path=DEFAULT_ONTOLOGY):
    """Load an optional user-supplied ontology; None uses ordinary token overlap."""
    if path is None:
        return {}, None
    path = Path(path).resolve()
    stat = path.stat()
    return _load(str(path), stat.st_mtime_ns, stat.st_size)


def entity_counts(words, trie):
    """Each token occurrence belongs to at most one longest phrase."""
    counts = Counter()
    index = 0
    while index < len(words):
        node, end, match = trie, index, None
        cursor = index
        while cursor < len(words) and words[cursor] in node:
            node = node[words[cursor]]
            cursor += 1
            if None in node:
                end, match = cursor, node[None]
        if match is None:
            index += 1
        else:
            counts[match] += 1
            index = end
    return counts


def ontology_metrics(left, right, ontology_path=DEFAULT_ONTOLOGY, entity_weight=DEFAULT_ENTITY_WEIGHT):
    validate_entity_weight(entity_weight)
    trie, _ = load_ontology(ontology_path)
    a, b = entity_counts(left, trie), entity_counts(right, trie)
    mass = lambda counts: sum(len(phrase.split()) * count for phrase, count in counts.items())
    total = len(left) + len(right)
    shared = sum((Counter(left) & Counter(right)).values())
    entity_total = mass(a) + mass(b)
    entity_shared = mass(a & b)
    weighted_total = total + (entity_weight - 1) * entity_total
    weighted_shared = shared + (entity_weight - 1) * entity_shared
    return {'ontology_weighted_overlap': 2 * weighted_shared / weighted_total if weighted_total else 0.0,
            'ontology_entity_agreement': 2 * entity_shared / entity_total if entity_total else None,
            'ontology_weighted_total': weighted_total,
            'ontology_weighted_shared': weighted_shared,
            'left_ontology_entities': dict(a), 'right_ontology_entities': dict(b),
            'unmatched_left_ontology_entities': dict(a - b),
            'unmatched_right_ontology_entities': dict(b - a)}
