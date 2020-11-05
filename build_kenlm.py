import shutil

import os
from tqdm import tqdm
from util import data_io

from erroneous_ngrams import calc_corrected_ngrams, tokenize, corrected_ngrams
from kenlm_arpa import convert_and_filter_topk, build_lm, ArpaArgs


if __name__ == "__main__":
    colab_asr_data = f"{os.environ['HOME']}/googledrive/data/asr_data"
    ngrams_file = "/tmp/ngrams.txt.gz"

    if not os.path.isfile(ngrams_file):
        ngrams = corrected_ngrams(f"{colab_asr_data}/results/train_kenlm_3_089_mp3")
        data_io.write_lines(ngrams_file, ngrams)
        data_io.write_lines("unique_ngrams.txt.gz", list(set(ngrams)))

    librispeech_lm_data = f"{os.environ['HOME']}/data/asr_data/ENGLISH/librispeech-lm-norm.txt.gz"

    for name, files in [
        ("vanilla", [librispeech_lm_data]),
        ("tedlium", [librispeech_lm_data]+10*[ngrams_file]),
    ]:
        cache_dir = f"kenlm_cache_{name}"

        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        data_lower, vocab_str = convert_and_filter_topk(files, cache_dir, 200_000)
        arpa_name = f"kenlm_{name}"
        arpa_args = ArpaArgs(
            f"{os.environ['HOME']}/code/CPP/kenlm/build/bin", arpa_name, 3, "20%", "0|8|9"
        )
        build_lm(
            arpa_args,
            data_lower,
            vocab_str,
        )

        os.system(
            f"gzip -c {arpa_name}/lm_filtered.arpa > {colab_asr_data}/ENGLISH/{arpa_name}.arpa.gz"
        )
