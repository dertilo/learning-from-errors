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

    def calc_ngram(s: str):
        tokens = s.split(" ")
        return [
            " ".join(tokens[k : (k + 1 + o)])
            for o in range(order)
            for k in range(len(tokens) - o)
        ]

    start = time()
    result = Counter(
        (
            " ".join(ref_ngram)
            for r, h in refs_hyps
            for ref_ngram, hyp_ngram in calc_aligned_ngram_tuples(
                tokenize(r), tokenize(h), order
            )
        )
    )
    print(result.most_common(10))
    print(f"took: {time()-start} seconds")

    start = time()
    b = db.from_sequence(refs_hyps)
    result = (
        b.map(
            lambda rh: calc_aligned_ngram_tuples(
                tokenize(rh[0]), tokenize(rh[1]), order
            )
        )
        .flatten()
        .map(lambda rh: " ".join(rh[0]))
        .frequencies()
        .topk(10, key=1)
    )
    print(result.compute())
    print(f"took: {time()-start} seconds")

"""
[('the', 122621), ('and', 76615), ('to', 60111), ('a', 58362), ('of', 57312), ('', 50024), ('that', 43012), ('i', 42357), ('in', 41354), ('is', 32955)]
took: 218.24797868728638 seconds
[('the', 122621), ('and', 76615), ('to', 60111), ('a', 58362), ('of', 57312), ('', 50024), ('that', 43012), ('i', 42357), ('in', 41354), ('is', 32955)]
took: 105.78709244728088 seconds
"""
