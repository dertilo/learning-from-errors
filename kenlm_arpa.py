import argparse
import gzip
import io
import os
import subprocess
from collections import Counter

# STOLEN FROM: https://github.com/mozilla/DeepSpeech/blob/master/data/lm/generate_lm.py
from tqdm import tqdm


def convert_and_filter_topk(args):
    """ Convert to lowercase, count word occurrences and save top-k words to a file """

    counter = Counter()
    data_lower = os.path.join(args.cache_dir, "lower.txt.gz")

    vocab_path = "vocab-{}.txt".format(args.top_k)
    vocab_path = os.path.join(args.cache_dir, vocab_path)


    if not os.path.isfile(data_lower) or not os.path.exists(vocab_path):
        print("\nConverting to lowercase and counting word occurrences ...")
        with io.TextIOWrapper(
            io.BufferedWriter(gzip.open(data_lower, "w+")), encoding="utf-8"
        ) as file_out:

            # Open the input file either from input.txt or input.txt.gz
            _, file_extension = os.path.splitext(args.input_txt)
            if file_extension == ".gz":
                file_in = io.TextIOWrapper(
                    io.BufferedReader(gzip.open(args.input_txt)), encoding="utf-8"
                )
            else:
                file_in = open(args.input_txt, encoding="utf-8")

            for line in tqdm(file_in):
                line_lower = line.lower()
                counter.update(line_lower.split())
                file_out.write(line_lower)

            file_in.close()

        # Save top-k words
        print("\nSaving top {} words ...".format(args.top_k))
        top_counter = counter.most_common(args.top_k)
        vocab_str = "\n".join(word for word, count in top_counter)

        with open(vocab_path, "w+") as file:
            file.write(vocab_str)

    with open(vocab_path, "r") as file:
        vocab_str = file.read()

    return data_lower, vocab_str


def build_lm(args, data_lower, vocab_str):
    print("\nCreating ARPA file ...")
    lm_path = os.path.join(args.output_dir, "lm.arpa")
    subargs = [
            os.path.join(args.kenlm_bins, "lmplz"),
            "--order",
            str(args.arpa_order),
            "--temp_prefix",
            args.output_dir,
            "--memory",
            args.max_arpa_memory,
            "--text",
            data_lower,
            "--arpa",
            lm_path,
            "--prune",
            *args.arpa_prune.split("|"),
        ]
    subprocess.check_call(subargs)

    # Filter LM using vocabulary of top-k words
    print("\nFiltering ARPA file using vocabulary of top-k words ...")
    filtered_path = os.path.join(args.output_dir, "lm_filtered.arpa")
    subprocess.run(
        [
            os.path.join(args.kenlm_bins, "filter"),
            "single",
            "model:{}".format(lm_path),
            filtered_path,
        ],
        input=vocab_str.encode("utf-8"),
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate lm.binary and top-k vocab for DeepSpeech."
    )
    parser.add_argument(
        "--input_txt",
        help="Path to a file.txt or file.txt.gz with sample sentences",
        type=str,
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
        "--kenlm_bins",
        help="File path to the KENLM binaries lmplz, filter and build_binary",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--arpa_order",
        help="Order of k-grams in ARPA-file generation",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max_arpa_memory",
        help="Maximum allowed memory usage for ARPA-file generation",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--arpa_prune",
        help="ARPA pruning parameters. Separate values with '|'",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)
    data_lower, vocab_str = convert_and_filter_topk(args)
    build_lm(args, data_lower, vocab_str)

if __name__ == "__main__":
    main()
