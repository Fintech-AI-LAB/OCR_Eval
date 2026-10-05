"""Command-line entry points for standalone OCR text evaluation."""
import argparse
import csv
import json
from pathlib import Path

from .batch import evaluate
from .page import score_page_details
from .benchmark import evaluate_benchmark, compare_benchmark_variants, consensus_family_sensitivity
from .profiles import load_json, load_profiles, extract_profile_predictions, make_reference_template


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def _write_benchmark(output, result):
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'results.json', result)
    with (output / 'ranking.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=('rank', 'method', 'critical_field_score'))
        writer.writeheader()
        writer.writerows(result['ranking'])
    def display(value):
        return 'N/A' if value is None else f'{100*value:.2f}%'
    lines = ['# Critical-field benchmark', '', result['meaning'], '',
             f"Annotation status: **{result['annotation_status']}**. Weighting: {result['settings']['weighting']}. "
             f"Split: {result['settings']['split']}.", '',
             '| Method | Critical score / 100 | Value precision | Value recall | Missing | Extra | All critical fields correct |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in result['ranking']:
        m = result['methods'][row['method']]['metrics']
        name = row['method'].replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {name} | {row['critical_field_score']:.2f} | {display(m['value_precision'])} | "
                     f"{display(m['value_recall'])} | {m['missing_fields']} | {m['extra_fields']} | "
                     f"{display(m['all_critical_fields_correct_document_rate'])} |")
    lines.extend(['', 'The critical score uses fixed reference fields and does not penalize extra predictions directly; '
                  'inspect precision and the document pass rate alongside it. Unreadable fields are excluded from value '
                  'accuracy and reported; documents with unreadable critical fields or required association targets '
                  'are ineligible for the document pass rate.', '',
                  'results.json contains per-field, per-document and per-template metrics, literal and typed matches, '
                  'relationships, localization, fixed-region CER, raw predictions, missing/extra evidence, '
                  'constraint violations and source/profile/reference hashes. Missing bounding boxes receive zero '
                  'localization credit. Missing fixed-region transcripts are scored as deletions.', ''])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate OCR consensus or accuracy against explicit references.')
    sub = parser.add_subparsers(dest='command', required=True)
    page = sub.add_parser('page', help='Score two or more OCR files for one physical page.')
    page.add_argument('files', nargs='+', type=Path)
    batch = sub.add_parser('batch', help='Rank page-aligned methods from a JSON file.')
    batch.add_argument('input', type=Path)
    batch.add_argument('--output', type=Path, default=Path('ocr-eval-results'))
    batch.add_argument('--weighting', choices=('document', 'page'), default='document')
    for command in (page, batch):
        command.add_argument('--ontology', type=Path, help='Optional replaceable ontology JSON file.')
        command.add_argument('--region-context-tokens', type=int, default=16)
        command.add_argument('--entity-weight', type=float, default=3.0)
        command.add_argument('--ontology-mode', choices=('regions', 'values', 'mentions'), default='regions',
                             help='With an ontology, compare text regions (default), exact values or legacy label mentions.')
        command.add_argument('--overlap-weight', type=float, default=1.0)
    benchmark = sub.add_parser('benchmark', help='Score fields and fixed-region transcripts against reviewed references.')
    benchmark.add_argument('--predictions', type=Path, required=True)
    variants = sub.add_parser('compare-variants', help='Compare prediction variants with fixed references and scoring.')
    variants.add_argument('--variant', action='append', required=True, metavar='NAME=PATH')
    for command in (benchmark, variants):
        command.add_argument('--profiles', type=Path, required=True)
        command.add_argument('--references', type=Path, required=True)
        command.add_argument('--split', choices=('train', 'validation', 'test', 'all'), default='test')
        command.add_argument('--weighting', choices=('document', 'field'), default='document')
        command.add_argument('--iou-threshold', type=float, default=0.5)
        command.add_argument('--output', type=Path, default=Path('ocr-benchmark-results'))
    extract = sub.add_parser('extract-profile', help='Extract explicit fields from aligned text without reference access.')
    extract.add_argument('input', type=Path)
    extract.add_argument('--document-profiles', type=Path, required=True, help='JSON mapping document IDs to profile IDs.')
    extract.add_argument('--families', type=Path, help='JSON mapping method names to actual engine families.')
    template = sub.add_parser('init-reference', help='Create a draft annotation inventory; never invent ground truth.')
    template.add_argument('manifest', type=Path)
    for command in (extract, template):
        command.add_argument('--profiles', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    sensitivity = sub.add_parser('family-sensitivity', help='Check existing consensus scores for engine-family dependence.')
    sensitivity.add_argument('pairs', type=Path)
    sensitivity.add_argument('--families', type=Path, required=True)
    sensitivity.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in {'benchmark', 'compare-variants'}:
            profiles, references = load_profiles(args.profiles), load_json(args.references)
            settings = dict(split=args.split, weighting=args.weighting, iou_threshold=args.iou_threshold)
            if args.command == 'benchmark':
                result = evaluate_benchmark(profiles, references, load_json(args.predictions), **settings)
                _write_benchmark(args.output, result)
                print(result['meaning'])
                for row in result['ranking']:
                    print(f"{row['rank']}. {row['method']}: {row['critical_field_score']:.2f}")
            else:
                inputs = {}
                for variant in args.variant:
                    name, separator, path = variant.partition('=')
                    if not separator or not name or name in inputs:
                        raise ValueError('Each variant must be a distinct NAME=PATH.')
                    inputs[name] = load_json(path)
                result = compare_benchmark_variants(profiles, references, inputs, **settings)
                args.output.mkdir(parents=True, exist_ok=True)
                _write_json(args.output / 'variants.json', result)
            print(f'Results: {args.output.resolve()}')
        elif args.command in {'extract-profile', 'init-reference', 'family-sensitivity'}:
            if args.command == 'extract-profile':
                result = extract_profile_predictions(load_json(args.input), load_profiles(args.profiles),
                    load_json(args.document_profiles), load_json(args.families) if args.families else None)
            elif args.command == 'init-reference':
                result = make_reference_template(load_profiles(args.profiles), load_json(args.manifest), args.manifest.parent)
            else:
                result = consensus_family_sensitivity(load_json(args.pairs), load_json(args.families))
            if args.command == 'init-reference' and args.output.exists():
                raise ValueError('Refusing to overwrite an existing annotation file; choose a new output path.')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(args.output, result)
            print(f'Written: {args.output.resolve()}')
        elif args.command == 'page':
            result = score_page_details(args.files, args.overlap_weight,
                                        ontology_path=args.ontology, entity_weight=args.entity_weight,
                                        ontology_mode=args.ontology_mode, region_context_tokens=args.region_context_tokens)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            data = json.loads(args.input.read_text(encoding='utf-8-sig'))
            result, pairs, volumes, normalized = evaluate(
                data, args.overlap_weight, args.weighting, args.ontology, args.entity_weight, args.ontology_mode, args.region_context_tokens)
            args.output.mkdir(parents=True, exist_ok=True)
            for name, value in (('results.json', result), ('page_differences.json', pairs),
                                ('text_volume.json', volumes), ('normalized_pages.json', normalized)):
                _write_json(args.output / name, value)
            with (args.output / 'ranking.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=('rank', 'model', 'consensus_score'))
                writer.writeheader()
                writer.writerows(result['ranking'])
            for row in result['ranking']:
                print(f"{row['rank']}. {row['model']}: {row['consensus_score']:.2f}")
            print(f'Results: {args.output.resolve()}')
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
