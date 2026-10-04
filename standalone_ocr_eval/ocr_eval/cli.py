"""Command-line entry points for standalone OCR text evaluation."""
import argparse
import csv
import json
from pathlib import Path

from .batch import evaluate
from .page import score_page_details


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Score OCR text consensus (not accuracy).')
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
    args = parser.parse_args(argv)
    try:
        if args.command == 'page':
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
