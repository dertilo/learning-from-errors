# based on  https://github.com/NVIDIA/NeMo/blob/main/examples/asr/speech_to_text_infer.py
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

from nemo.collections.asr.metrics.wer import WER, word_error_rate
from nemo.collections.asr.models import EncDecCTCModel
from torch.nn import Softmax
from tqdm import tqdm
from util import data_io
import nemo.collections.asr as nemo_asr
import numpy as np
import os

from prepare_arpa import prepare_arpa_file

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


can_gpu = torch.cuda.is_available()


def beamsearch_forward(beam_search_lm, log_probs, log_probs_length):
    bs = beam_search_lm
    probs = torch.exp(log_probs)
    probs_list = []
    for i, prob in enumerate(probs):
        probs_list.append(prob[: log_probs_length[i], :])
    res = bs.beam_search_func(
        probs_list,
        bs.vocab,
        beam_size=bs.beam_width,
        num_processes=bs.num_cpus,
        ext_scoring_func=bs.scorer,
        cutoff_prob=bs.cutoff_prob,
        cutoff_top_n=bs.cutoff_top_n,
    )
    return [r[0][1] for r in res]


def generate_ref_hyps(asr_model, search, arpa):

    if can_gpu:
        asr_model = asr_model.cuda()
        print("USING GPU!")

    asr_model.eval()
    labels_map = dict(
        [
            (i, asr_model.decoder.vocabulary[i])
            for i in range(len(asr_model.decoder.vocabulary))
        ]
    )
    wer = WER(vocabulary=asr_model.decoder.vocabulary)

    if search == "kenlm" or search == "beamsearch":
        arpa_file = prepare_arpa_file(arpa)
        lm_path = arpa_file if search == "kenlm" else None

        beamsearcher = nemo_asr.modules.BeamSearchDecoderWithLM(
            vocab=list(asr_model.cfg.decoder.params.vocabulary),
            beam_width=16,
            alpha=2,
            beta=1.5,
            lm_path=lm_path,
            num_cpus=max(os.cpu_count(), 1),
            input_tensor=True,
        )

    for test_batch in tqdm(asr_model.test_dataloader()):
        if can_gpu:
            test_batch = [x.cuda() for x in test_batch]
        with autocast():
            log_probs, encoded_len, greedy_predictions = asr_model(
                input_signal=test_batch[0], input_signal_length=test_batch[1]
            )
        if search == "greedy":
            decoded = wer.ctc_decoder_predictions_tensor(greedy_predictions)
        else:
            decoded = beamsearch_forward(
                beamsearcher, log_probs=log_probs, log_probs_length=encoded_len
            )

        for batch_ind, hyp in enumerate(decoded):
            reference = "".join(
                [labels_map[c] for c in test_batch[2][batch_ind].cpu().detach().numpy()]
            )
            yield reference, hyp


def prepare_manifest(corpora_dir="/content/corpora"):

    manifest = "manifest.jsonl"

    g = (
        {
            "audio_filepath": f"{str(f).replace(f.name, '')}{d['audio_file']}",
            "duration": d["duration"],
            "text": d["text"],
        }
        for f in list(Path(corpora_dir).rglob("manifest.jsonl.gz"))
        for d in data_io.read_jsonl(str(f))
    )
    data_io.write_jsonl(manifest, g)
    return manifest


def main():
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--asr_model", type=str, default="QuartzNet5x5LS-En", help="Pass: 'QuartzNet15x5Base-En'")
    parser.add_argument("--corpora_dir", type=str, default="/tmp/corpora",help="directory containing corpora")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--normalize_text", default=True, type=bool, help="Normalize transcripts or not. Set to False for non-English.")
    parser.add_argument("--search", default="greedy", type=str, choices=['greedy', 'beamsearch', 'kenlm'], help="greedy or beamsearch or beamsearch+KenLM")
    parser.add_argument("--arpa", default='3-gram.pruned.1e-7.arpa', type=str, help="arpa file")
    parser.add_argument("--refs", default='refs.txt', type=str)
    parser.add_argument("--hyps", default='hyps.txt', type=str)
    parser.add_argument("--stats", default='stats.json', type=str)

    # fmt: on

    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if args.asr_model.endswith(".nemo"):
        print(f"Using local ASR model from {args.asr_model}")
        asr_model = EncDecCTCModel.restore_from(restore_path=args.asr_model)
    else:
        print(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = EncDecCTCModel.from_pretrained(model_name=args.asr_model)

    manifest = prepare_manifest(args.corpora_dir)
    asr_model.setup_test_data(
        test_data_config={
            "sample_rate": 16000,
            "manifest_filepath": manifest,
            "labels": asr_model.decoder.vocabulary,
            "batch_size": args.batch_size,
            "normalize_transcripts": args.normalize_text,
        }
    )

    refs_hyps = list(generate_ref_hyps(asr_model, args.search, args.arpa))
    hypotheses, references = [list(k) for k in zip(*refs_hyps)]

    data_io.write_jsonl(args.refs, references)
    data_io.write_jsonl(args.hyps, hypotheses)

    wer_value = word_error_rate(hypotheses=hypotheses, references=references)
    sys.stdout.flush()
    stats = {
        "wer":wer_value,
        "args":args.__dict__,
    }
    data_io.write_json(args.stats,stats)
    print(f"Got WER of {wer_value}")


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
