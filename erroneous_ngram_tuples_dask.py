from pprint import pprint

import json
from collections import Counter

from time import time

import dask.bag as db
from tqdm import tqdm
from util import data_io
from dask.distributed import Client, progress

from alignment import calc_aligned_ngram_tuples

if __name__ == "__main__":
    client = Client()
    print(client)

    refs_hyps_dir = "/tmp/train_kenlm_3_089_mp3"
    refs = data_io.read_lines(f"{refs_hyps_dir}/refs.txt.gz")
    hyps = data_io.read_lines(f"{refs_hyps_dir}/hyps.txt.gz")
    refs_hyps = list((ref, hyp) for ref, hyp in zip(refs, hyps))

    tokenize = lambda s: s.split(" ")
    order = 3

    start = time()
    aligned_ngrams = (
        db.from_sequence(refs_hyps, npartitions=4 * 4)
        .map(
            lambda rh: calc_aligned_ngram_tuples(
                tokenize(rh[0]), tokenize(rh[1]), order
            )
        )
        .flatten()
        .map(lambda rh: (" ".join(rh[0]), " ".join(rh[1])))
    )

    def error_rate(ref,hyp_counts:Counter):
        overall_num_erros = sum(v for k, v in hyp_counts.items() if ref != k)
        num_correct = hyp_counts[ref]
        return overall_num_erros / (1+num_correct)

    result = aligned_ngrams.foldby(
        lambda rh: rh[0],
        lambda total, x: total + Counter([x[1]]),
        initial=Counter(),
        combine=lambda x, y: x + y,
    ).topk(1000, lambda kc:error_rate(*kc))
    counts = result.map(lambda kc: (kc[0], dict(kc[1].most_common(5)))).compute()
    data_io.write_jsonl("ngram_counts.jsonl", counts)

    # pprint(result.compute()) #topk(10,key=lambda )
    # aligned_ngrams.filter(lambda rh: rh[0] != rh[1]).map(json.dumps).to_textfiles(
    #     "processed/erroneous_ngrams_*.jsonl.gz"
    # )
    print(f"took: {time()-start} seconds")
