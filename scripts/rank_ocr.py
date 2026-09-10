#!/usr/bin/env python3
"""Reference-free text consensus ranking from saved pairwise page similarities."""
import collections
import itertools
import json
import random
import statistics
from pathlib import Path
from ocr_common import ROOT, write_json

OUT = ROOT / 'output' / 'performance'
ENGINES = ['mistral', 'openai-6', 'fable-5-1', 'mineru']
FAMILY = {'mistral':'mistral', 'mineru':'mineru', 'openai-6':'tesseract', 'fable-5-1':'tesseract'}


def calculate(rows, overlap_weight=0.5, family_adjusted=False):
    by_page = collections.defaultdict(dict)
    for r in rows:
        by_page[r['document'],r['page']][frozenset((r['left'],r['right']))] = (
            overlap_weight*r['token_overlap']+(1-overlap_weight)*r['sequence_similarity'])
    result = collections.defaultdict(lambda: collections.defaultdict(list))
    for (doc,page),pairs in sorted(by_page.items()):
        if len(pairs)!=6:
            raise ValueError(f'Expected six comparisons at {doc} page {page}')
        for engine in ENGINES:
            if family_adjusted:
                groups=collections.defaultdict(list)
                for peer in ENGINES:
                    if FAMILY[peer]!=FAMILY[engine]:
                        groups[FAMILY[peer]].append(pairs[frozenset((engine,peer))])
                score=statistics.mean(statistics.mean(v) for v in groups.values())
            else:
                score=statistics.mean(pairs[frozenset((engine,peer))] for peer in ENGINES if peer!=engine)
            result[doc][engine].append(score)
    return dict(result)


def aggregate(page_scores, page_weighted=False):
    return {e:statistics.mean([v for doc in page_scores.values() for v in doc[e]])
            if page_weighted else statistics.mean(statistics.mean(doc[e]) for doc in page_scores.values())
            for e in ENGINES}


def ordered(scores):
    return sorted(scores, key=lambda e: (-scores[e],e))


def main():
    rows=json.loads((OUT/'agreement.json').read_text())
    primary=calculate(rows)
    score=aggregate(primary)
    per_doc={d:{e:statistics.mean(v[e]) for e in ENGINES} for d,v in primary.items()}
    variants={
        'Primary: equal documents, 50% overlap / 50% sequence':score,
        'Equal pages instead of equal documents':aggregate(primary,True),
        'Overlap only, equal documents':aggregate(calculate(rows,1)),
        'Sequence only, equal documents':aggregate(calculate(rows,0)),
        'Exclude same-family peers; equal peer-family weight':aggregate(calculate(rows,family_adjusted=True)),
    }
    # Joint page resampling keeps all four methods paired and each document represented.
    rng=random.Random(20260910);wins=collections.Counter();samples={e:[] for e in ENGINES}
    for _ in range(2000):
        totals={e:[] for e in ENGINES}
        for doc,v in primary.items():
            indices=[rng.randrange(len(v[ENGINES[0]])) for _ in v[ENGINES[0]]]
            for e in ENGINES:totals[e].append(statistics.mean(v[e][i] for i in indices))
        current={e:statistics.mean(totals[e]) for e in ENGINES}
        best=max(current.values());tied=[e for e in ENGINES if abs(current[e]-best)<1e-12]
        for e in tied:wins[e]+=1/len(tied)
        for e in ENGINES:samples[e].append(current[e])
    rank=[]
    for n,e in enumerate(ordered(score),1):
        values=sorted(samples[e])
        rank.append({'rank':n,'method':e,'consensus_score':100*score[e],
                     'page_resample_first_share':wins[e]/2000,
                     'page_resample_95_percentile_range':[100*values[49],100*values[1949]]})
    result={'algorithm':'Equal-document weighted consensus centrality',
            'pair_score':'0.5 token overlap + 0.5 symmetric token-sequence similarity',
            'aggregation':'Mean of three peers per page, mean pages per document, mean of three documents',
            'meaning':'Similarity to peer outputs, not correctness or OCR accuracy',
            'ranking':rank,'per_document_scores':per_doc,'sensitivity_scores':variants,
            'resampling':{'repetitions':2000,'seed':20260910,
                          'meaning':'Descriptive stability under within-document page resampling; not probability of being correct'}}
    write_json(OUT/'ranking.json',result)
    lines=['# Text-consensus ranking', '',
           f'**{ordered(score)[0]} ranks first under the primary consensus algorithm below.** This is the best consensus match among the four saved text outputs, not a demonstrated accuracy winner. No source transcription is treated as ground truth.', '',
           '## Algorithm', '',
           'For each of the 48 pages, create a four-node similarity graph. The weight between two methods is 50% token overlap plus 50% symmetric token-sequence similarity. A method’s page score is its mean edge weight to the other three methods (weighted-degree centrality). Average its page scores within each document, then average the three document scores equally. Multiply by 100 for display.', '',
           'Equal document weighting gives the 4-page board resolution, 14-page agreement and 30-page trade document the same influence. The 50/50 mixture balances shared words and word order; it is a stated judgment, not a trained or validated accuracy model. Tokenization ignores case and punctuation. Numeric spelling is reflected in tokens, but punctuation-only differences are not scored. Structure, field schemas, table formatting, speed and cost are excluded.', '',
           '| Rank | Saved method | Consensus score / 100 | First in page resamples |',
           '|---:|---|---:|---:|']
    for r in rank:lines.append(f"| {r['rank']} | {r['method']} | {r['consensus_score']:.2f} | {r['page_resample_first_share']:.1%} |")
    lines += ['', 'The resampling column uses 2,000 paired page resamples within each document. It measures ranking stability for these documents only, not a probability that a method is correct. Pages and documents may be correlated; three documents are insufficient for broad generalization.', '',
              '## By document', '', '| Method | Agreement | Board resolution | Trade |','|---|---:|---:|---:|']
    for e in ordered(score):lines.append(f'| {e} | '+' | '.join(f'{100*per_doc[d][e]:.2f}' for d in ['csa','board','trade'])+' |')
    lines += ['', '## Sensitivity checks', '', '| Scoring variant | Ranking, highest first |','|---|---|']
    for label,v in variants.items():lines.append('| '+label+' | '+' > '.join(f'{e} ({100*v[e]:.2f})' for e in ordered(v))+' |')
    lines += ['', 'The family-adjusted variant excludes a candidate’s same-family peer and gives each remaining OCR family equal weight. For example, Mistral compares equally with MinerU and the average of the two Tesseract-labeled outputs. This reduces duplicate voting but is not a statistical correction for all shared errors.', '',
              '## What “best” means here', '',
              '- A high score means that a text is central to the other saved outputs. It does not prove that the text is complete, faithful, or free of shared errors.',
              '- A unique correct reading can be penalized; shared incorrect readings can be rewarded. There is no justified absolute accuracy ranking without a reference or independent verification.',
              '- The openai-6 and fable-5-1 artifact metadata identifies Tesseract plus visual review. Their folder names cannot establish GPT-6 or Fable model performance. Fable’s raw full_page_text is used; its separately reviewed fields are excluded.',
              '- Changes in input format, resolution and post-processing remain confounds. Ranking the saved workflows does not isolate the underlying model.', '',
              f'Use {ordered(score)[0]} as the **consensus representative** for this collection, and inspect low-agreement pages before choosing any text as authoritative. See [text comparison](report.md) and [page differences](pages.html).', '',
              'Reproduce: `conda run --no-capture-output -n myenv3.13 python scripts/evaluate_outputs.py` followed by `conda run --no-capture-output -n myenv3.13 python scripts/rank_ocr.py`.',
              'Machine-readable results: [ranking.json](ranking.json).']
    (OUT/'ranking.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
