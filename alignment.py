from __future__ import print_function

from collections import Counter

from pprint import pprint

import argparse
import logging
import sys

logger = logging.getLogger(__name__)
verbose_level = 0

# fmt: off
def smith_waterman_alignment(ref, hyp, similarity_score_function,
                             del_score, ins_score,
                             eps_symbol="<eps>", align_full_hyp=True):
    """
    stolen from kaldi; see egs/wsj/s5/steps/cleanup/internal/align_ctm_ref.py

    Does Smith-Waterman alignment of reference sequence and hypothesis
    sequence.
    This is a special case of the Smith-Waterman alignment that assumes that
    the deletion and insertion costs are linear with number of incorrect words.

    If align_full_hyp is True, then the traceback of the alignment
    is started at the end of the hypothesis. This is when we want the
    reference that aligns with the full hypothesis.
    This differs from the normal Smith-Waterman alignment, where the traceback
    is from the highest score in the alignment score matrix. This
    can be obtained by setting align_full_hyp as False. This gets only the
    sub-sequence of the hypothesis that best matches with a
    sub-sequence of the reference.

    Returns a list of tuples where each tuple has the format:
        (ref_word, hyp_word, ref_word_from_index, hyp_word_from_index,
         ref_word_to_index, hyp_word_to_index)
    """
    output = []

    ref_len = len(ref)
    hyp_len = len(hyp)

    bp = [[] for x in range(ref_len+1)]

    # Score matrix of size (ref_len + 1) x (hyp_len + 1)
    # The index m, n in this matrix corresponds to the score
    # of the best matching sub-sequence pair between reference and hypothesis
    # ending with the reference word ref[m-1] and hypothesis word hyp[n-1].
    # If align_full_hyp is True, then the hypothesis sub-sequence is from
    # the 0th word i.e. hyp[0].
    H = [[] for x in range(ref_len+1)]

    for ref_index in range(ref_len+1):
        if align_full_hyp:
            H[ref_index] = [-(hyp_len+2) for x in range(hyp_len+1)]
            H[ref_index][0] = 0
        else:
            H[ref_index] = [0 for x in range(hyp_len+1)]
        bp[ref_index] = [(0, 0) for x in range(hyp_len+1)]

        if align_full_hyp and ref_index == 0:
            for hyp_index in range(1, hyp_len+1):
                H[0][hyp_index] = H[0][hyp_index-1] + ins_score
                bp[ref_index][hyp_index] = (ref_index, hyp_index-1)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4}"
                    "".format(ref_index, hyp_index-1, ref_index, hyp_index,
                              H[ref_index][hyp_index]))

    max_score = -float("inf")
    max_score_element = (0, 0)

    for ref_index in range(1, ref_len+1):     # Reference
        for hyp_index in range(1, hyp_len+1):     # Hypothesis
            sub_or_ok = (H[ref_index-1][hyp_index-1]
                         + similarity_score_function(ref[ref_index-1],
                                                     hyp[hyp_index-1]))

            if ((not align_full_hyp and sub_or_ok > 0)
                    or (align_full_hyp
                        and sub_or_ok >= H[ref_index][hyp_index])):
                H[ref_index][hyp_index] = sub_or_ok
                bp[ref_index][hyp_index] = (ref_index-1, hyp_index-1)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4} ({5},{6})"
                    "".format(ref_index-1, hyp_index-1, ref_index, hyp_index,
                              H[ref_index][hyp_index],
                              ref[ref_index-1], hyp[hyp_index-1]))

            if H[ref_index-1][hyp_index] + del_score > H[ref_index][hyp_index]:
                H[ref_index][hyp_index] = H[ref_index-1][hyp_index] + del_score
                bp[ref_index][hyp_index] = (ref_index-1, hyp_index)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4}"
                    "".format(ref_index-1, hyp_index, ref_index, hyp_index,
                              H[ref_index][hyp_index]))

            if H[ref_index][hyp_index-1] + ins_score > H[ref_index][hyp_index]:
                H[ref_index][hyp_index] = H[ref_index][hyp_index-1] + ins_score
                bp[ref_index][hyp_index] = (ref_index, hyp_index-1)
                logger.debug(
                    "({0},{1}) -> ({2},{3}): {4}"
                    "".format(ref_index, hyp_index-1, ref_index, hyp_index,
                              H[ref_index][hyp_index]))

            #if hyp_index == hyp_len and H[ref_index][hyp_index] >= max_score:
            if ((not align_full_hyp or hyp_index == hyp_len)
                    and H[ref_index][hyp_index] >= max_score):
                max_score = H[ref_index][hyp_index]
                max_score_element = (ref_index, hyp_index)

    ref_index, hyp_index = max_score_element
    score = max_score
    logger.debug("Alignment score: %s for (%d, %d)",
                 score, ref_index, hyp_index)

    while ((not align_full_hyp and score >= 0)
           or (align_full_hyp and hyp_index > 0)):
        try:
            prev_ref_index, prev_hyp_index = bp[ref_index][hyp_index]
            if ((prev_ref_index, prev_hyp_index) == (ref_index, hyp_index)
                    or (prev_ref_index, prev_hyp_index) == (0, 0)):
                score = H[ref_index][hyp_index]
                if score != 0:
                    ref_word = ref[ref_index-1] if ref_index > 0 else eps_symbol
                    hyp_word = hyp[hyp_index-1] if hyp_index > 0 else eps_symbol
                    output.append((ref_word, hyp_word, prev_ref_index,
                        prev_hyp_index, ref_index, hyp_index))

                    ref_index, hyp_index = (prev_ref_index, prev_hyp_index)
                    score = H[ref_index][hyp_index]
                break

            if (ref_index == prev_ref_index + 1
                    and hyp_index == prev_hyp_index + 1):
                # Substitution or correct
                output.append(
                    (ref[ref_index-1] if ref_index > 0 else eps_symbol,
                     hyp[hyp_index-1] if hyp_index > 0 else eps_symbol,
                     prev_ref_index, prev_hyp_index, ref_index, hyp_index))
            elif (prev_hyp_index == hyp_index):
                # Deletion
                assert prev_ref_index == ref_index - 1
                output.append(
                    (ref[ref_index-1] if ref_index > 0 else eps_symbol,
                     eps_symbol,
                     prev_ref_index, prev_hyp_index, ref_index, hyp_index))
            elif (prev_ref_index == ref_index):
                # Insertion
                assert prev_hyp_index == hyp_index - 1
                output.append(
                    (eps_symbol,
                     hyp[hyp_index-1] if hyp_index > 0 else eps_symbol,
                     prev_ref_index, prev_hyp_index, ref_index, hyp_index))
            else:
                raise RuntimeError


            ref_index, hyp_index = (prev_ref_index, prev_hyp_index)
            score = H[ref_index][hyp_index]
        except Exception:
            logger.error("Unexpected entry (%d,%d) -> (%d,%d), %s, %s",
                         prev_ref_index, prev_hyp_index, ref_index, hyp_index,
                         ref[prev_ref_index], hyp[prev_hyp_index])
            raise RuntimeError("Unexpected result: Bug in code!!")

    assert (align_full_hyp or score == 0)

    output.reverse()

    if verbose_level > 2:
        for ref_index in range(ref_len+1):
            for hyp_index in range(hyp_len+1):
                print ("{0} ".format(H[ref_index][hyp_index]), end='',
                       file=sys.stderr)
            print ("", file=sys.stderr)

    logger.debug("Aligned output:")
    logger.debug("  -  ".join(["({0},{1})".format(x[4], x[5])
                               for x in output]))
    logger.debug("REF: ")
    logger.debug("    ".join(str(x[0]) for x in output))
    logger.debug("HYP:")
    logger.debug("    ".join(str(x[1]) for x in output))

    return (output, max_score)

# fmt: on


def get_edit_type(ref, hyp, eps="-"):
    if ref != hyp and not (ref == eps or hyp == eps):
        et = "sub"
    elif ref != hyp and ref == eps:
        et = "ins"
    elif ref != hyp and hyp == eps:
        et = "del"
    else:
        et = "cor"

    return et


def align_and_calc_edit_types(ref_tok, hyp_tok):

    eps = "|"
    output, score = smith_waterman_alignment(
        ref_tok,
        hyp_tok,
        similarity_score_function=lambda x, y: 2 if (x == y) else -1,
        del_score=-1,
        ins_score=-1,
        eps_symbol=eps,
        align_full_hyp=True,
    )

    ets = [get_edit_type(r, h, eps) for r, h, *_ in output]
    return ets


if __name__ == "__main__":
    from util import data_io

    refs = list(data_io.read_lines("3-gram.pruned.1e-7.arpa_wav_refs.txt"))
    hyps = list(data_io.read_lines("3-gram.pruned.1e-7.arpa_wav_hyps.txt"))
    pprint(
        Counter(
            e
            for r, h in zip(refs, hyps)
            for e in align_and_calc_edit_types(r.split(), h.split())
        )
    )
# if __name__ == '__main__':
#     hyp = "Thee cad i blac"
#     ref = "The cat is black"
#
#     verbose = 3
#     eps = '|'
#
#     output, score = smith_waterman_alignment(
#         ref, hyp, similarity_score_function=lambda x, y: 2 if (x == y) else -1,
#         del_score=-1, ins_score=-1, eps_symbol=eps, align_full_hyp=True)
#     print("ref: "+"".join(x[0] for x in output))
#     print("hyp: "+"".join(x[1] for x in output))
#
#     print(output)
#     print([f"{r}->{h}: {get_edit_type(r, h, eps)}" for r,h,*_ in output])
