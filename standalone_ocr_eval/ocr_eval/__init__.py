"""OCR text consensus scoring without a ground-truth transcription.

The optional ontology_path accepts a JSON file with a nonempty ``concepts``
mapping. With an ontology, the default score compares ontology-anchored text regions. Omitting it
uses unweighted token overlap; ontology_mode="mentions" selects legacy weighting.
"""

from .batch import evaluate
from .ontology import load_ontology, ontology_metrics
from .field_values import extract_field_values, field_value_metrics
from .regions import extract_ontology_regions, region_metrics
from .page import score_models, score_page, score_page_details
from .text import ALGORITHM_VERSION, compare_page, normalize_texts
from .benchmark import evaluate_benchmark, compare_benchmark_variants, consensus_family_sensitivity
from .profiles import load_profiles, extract_profile_predictions, make_reference_template

__all__ = [
    'ALGORITHM_VERSION', 'compare_page', 'evaluate', 'load_ontology',
    'normalize_texts', 'ontology_metrics', 'extract_field_values', 'field_value_metrics',
    'extract_ontology_regions', 'region_metrics', 'score_models', 'score_page', 'score_page_details',
    'evaluate_benchmark', 'compare_benchmark_variants', 'consensus_family_sensitivity',
    'load_profiles', 'extract_profile_predictions', 'make_reference_template',
]
