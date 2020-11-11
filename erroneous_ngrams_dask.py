from collections import Counter

from time import time

import dask.bag as db
from tqdm import tqdm
from util import data_io
from dask.distributed import Client, progress


if __name__ == "__main__":
    client = Client()
    print(client)

    refs_hyps_dir = "/tmp/train_kenlm_3_089_mp3"
    refs = data_io.read_lines(f"{refs_hyps_dir}/refs.txt.gz")
    hyps = data_io.read_lines(f"{refs_hyps_dir}/hyps.txt.gz")
    refs_hyps = list((ref, hyp) for ref, hyp in tqdm(zip(refs, hyps)))

    order = 3
    def calc_ngram(s:str):
        tokens = s.split(" ")
        return [" ".join(tokens[k:(k+1+o)]) for o in range(order) for k in range(len(tokens) - o)]

    start = time()
    result = Counter((ngram for r,h in refs_hyps for ngram in calc_ngram(r)))
    print(result.most_common(10))
    print(f"took: {time()-start} seconds")

    start = time()
    b = db.from_sequence([r for r,h in refs_hyps], npartitions=2)
    result = b.map(calc_ngram).flatten().frequencies().topk(10,key=1)
    print(result.compute())
    print(f"took: {time()-start} seconds")

"""
90979it [00:00, 251896.41it/s]
[('the', 108862), ('and', 78528), ('to', 63480), ('of', 60309), ('a', 54473), ('that', 45690), ('in', 40159), ('i', 36119), ('is', 34201), ('you', 33433)]
took: 4.292733907699585 seconds
[('the', 108862), ('and', 78528), ('to', 63480), ('of', 60309), ('a', 54473), ('that', 45690), ('in', 40159), ('i', 36119), ('is', 34201), ('you', 33433)]
took: 4.792858839035034 seconds
"""