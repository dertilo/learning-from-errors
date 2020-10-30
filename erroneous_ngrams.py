from typing import List, Tuple

from util import data_io

from alignment import smith_waterman_alignment, get_edit_type


import re

def tokenize(
    text, pattern=r"(?u)\b\w\w+\b"
) -> List[str]:  # pattern stolen from scikit-learn
    # TODO(tilo) this should actually be same tokenizer as used in ASR-Pipeline and KenLM
    # return [(m.start(),m.end(),m.group()) for m in re.finditer(pattern, text)]
    return text.split(" ")  # I suppose librispeech-data is very stupid


def calc_corrected_ngrams(ref_tok, hyp_tok, order=3):
    output, score = smith_waterman_alignment(
        ref_tok,
        hyp_tok,
        similarity_score_function=lambda x, y: 2 if (x == y) else -1,
        del_score=-1,
        ins_score=-1,
        eps_symbol=eps,
        align_full_hyp=True,
    )

    is_edited = [get_edit_type(r, h, eps) != "cor" for r, h, *_ in output]

    return [
        [x for x, *_ in output[(k - order) : k]]
        for k in range(order, len(output) + 1)
        if any(is_edited[(k - order) : k])
    ]


if __name__ == "__main__":
    eps = "|"
    refs = list(data_io.read_lines("3-gram.pruned.1e-7.arpa_wav_refs.txt"))
    hyps = list(data_io.read_lines("3-gram.pruned.1e-7.arpa_wav_hyps.txt"))

    order = 3
    g = (
        " ".join(ngram)
        for ref, hyp in zip(refs, hyps)
        for ngram in calc_corrected_ngrams(tokenize(ref), tokenize(hyp), order)
    )

    data_io.write_lines("ngrams.txt", g)
