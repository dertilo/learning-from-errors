from typing import NamedTuple, List

import argparse
import gzip
import io
import os
import subprocess
from collections import Counter

# based on: https://github.com/mozilla/DeepSpeech/blob/master/data/lm/generate_lm.py
from tqdm import tqdm


def process_file(counter, file_out, input_file):
    _, file_extension = os.path.splitext(input_file)
    if file_extension == ".gz":
        file_in = io.TextIOWrapper(
            io.BufferedReader(gzip.open(input_file)), encoding="utf-8"
        )
    else:
        file_in = open(input_file, encoding="utf-8")
    for line in tqdm(file_in):
        line_lower = line.lower()
        counter.update(line_lower.split())
        file_out.write(line_lower)
    file_in.close()


def convert_and_filter_topk(input_txt_files:List[str], cache_dir, top_k):
    """ Convert to lowercase, count word occurrences and save top-k words to a file """

    counter = Counter()
    data_lower = os.path.join(cache_dir, "lower.txt.gz")

    vocab_path = "vocab-{}.txt".format(top_k)
    vocab_path = os.path.join(cache_dir, vocab_path)

    if not os.path.isfile(data_lower) or not os.path.exists(vocab_path):
        print("\nConverting to lowercase and counting word occurrences ...")
        with io.TextIOWrapper(
            io.BufferedWriter(gzip.open(data_lower, "w+")), encoding="utf-8"
        ) as file_out:

            # Open the input file either from input.txt or input.txt.gz

            for input_file in input_txt_files:
                process_file(counter, file_out, input_file)

        # Save top-k words
        print("\nSaving top {} words ...".format(top_k))
        top_counter = counter.most_common(top_k)
        vocab_str = "\n".join(word for word, count in top_counter)

        with open(vocab_path, "w+") as file:
            file.write(vocab_str)

    with open(vocab_path, "r") as file:
        vocab_str = file.read()

    return data_lower, vocab_str


class ArpaArgs(NamedTuple):
    kenlm_bin: str
    output_dir: str = "kenlm"
    order: int = 3
    max_memory: str = "20%"
    prune: str = "0|8|9"


def build_lm(args: ArpaArgs, data_lower, vocab_str):
    print("\nCreating ARPA file ...")
    os.makedirs(args.output_dir,exist_ok=True)
    lm_path = os.path.join(args.output_dir, "lm.arpa")
    subargs = [
        os.path.join(args.kenlm_bin, "lmplz"),
        "--order",
        str(args.order),
        "--temp_prefix",
        args.output_dir,
        "--memory",
        args.max_memory,
        "--text",
        data_lower,
        "--arpa",
        lm_path,
        "--prune",
        *args.prune.split("|"),
    ]
    subprocess.check_call(subargs)

    # Filter LM using vocabulary of top-k words
    print("\nFiltering ARPA file using vocabulary of top-k words ...")
    filtered_path = os.path.join(args.output_dir, "lm_filtered.arpa")
    subprocess.run(
        [
            os.path.join(args.kenlm_bin, "filter"),
            "single",
            "model:{}".format(lm_path),
            filtered_path,
        ],
        input=vocab_str.encode("utf-8"),
        check=True,
    )


def main():
    # fmt: off
    parser = argparse.ArgumentParser(
        description="Generate lm.binary and top-k vocab for DeepSpeech."
    )
    parser.add_argument(
        "--input_txt",
        help="Path to a file.txt or file.txt.gz with sample sentences",
        type=str,
        nargs='+',
        required=True,
    )
    parser.add_argument(
        "--output_dir", help="Directory path for the output", type=str, required=True
    )
    parser.add_argument(
        "--cache_dir", default="kenlm_cache", type=str
    )

    parser.add_argument(
        "--top_k",
        help="Use top_k most frequent words for the vocab.txt file. These will be used to filter the ARPA file.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--kenlm_bin",
        help="File path to the KENLM binaries lmplz, filter and build_binary",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--order",
        help="Order of k-grams in ARPA-file generation",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max_memory",
        help="Maximum allowed memory usage for ARPA-file generation",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--prune",
        help="ARPA pruning parameters. Separate values with '|'",
        type=str,
        required=True,
    )
    # fmt: on

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data_lower, vocab_str = convert_and_filter_topk(
        args.input_txt, args.cache_dir, args.top_k
    )
    arpa_args = ArpaArgs(
        args.kenlm_bin, args.output_dir, args.order, args.max_memory, args.prune
    )
    build_lm(
        arpa_args,
        data_lower,
        vocab_str,
    )


if __name__ == "__main__":
    main()
