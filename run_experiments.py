from pprint import pprint

import argparse

from run_batch_inference import batch_inference

if __name__ == "__main__":
    limit = 200

    opelslr_1e7 = "3-gram.pruned.1e-7.arpa"
    stats = []

    for arpa, format in [
        # (opelslr_1e7, "wav"), (opelslr_1e7, "mp3"),
        ("lm_0_7.arpa","wav"), ("lm_0_7.arpa","mp3")
                         ]:
        params = {
            "asr_model": "QuartzNet5x5LS-En",
            "corpora_dir": f"/home/tilo/data/asr_data/ENGLISH/dev-other_processed_{format}",
            "batch_size": 32,
            "normalize_text": True,
            "search": "kenlm",
            "arpa": arpa,
            "name": "1e7_mp3",
            "limit": 200,
        }

        args = argparse.Namespace(**params)
        stats.append(batch_inference(args))

    pprint(stats)