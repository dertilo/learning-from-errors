# Learning From Errors
**errors made by a speech recognition system**  

__overall idea__: by "finetuning" an ngram-based Language Model one should be able to counteract errors that are specific to the acoustic model

## 1. "generating" errors
1. take pretrained NeMo __QuartzNet__ ([QuartzNet5x5LS-En](https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels))
2. train [KenLM](https://github.com/kpu/kenlm) on [__librispeech-lm-data__](http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz); I suppose thats how LM models at [openslr](http://www.openslr.org/11/) where created
    -> receive __librispeech-KenLM__
3. use __QuartzNet__ + __librispeech-KenLM__ to predict on [TEDLIUMv2 trainset](https://www.openslr.org/19/)
    -> receive ~90k __correction tuples__ `(hypothesis,reference)`
4. use [smith-waterman-algorithm](alignment.py) (stolen from kaldi) to align hypothesis (predicitons) and references
    -> looks like
```shell script
hyp = "hee cad i blac"
ref = "I think the cat is black"

alignment:
ref: I think th|e cat is black
hyp: |||||||||hee cad i| blac|
```
4. [analyse errors](erroneous_ngram_tuples_dask.py)
* TED-talks contain `missus`: 98 times the __QuartzNet__ + __librispeech-KenLM__ understood `this` instead of `missus`
```shell script
["missus", {"this": 98, "is": 98, "this is": 76, "a": 9, "and": 7}]
```
* TED-talks preferes two word `per cent`
```shell script
["per cent", {"percent": 1083, "percent of": 17, "percent oil": 3, "percent that": 2, "forty percent": 2}]
```
* 104 times __QuartzNet__ + __librispeech-KenLM__ inserts an `and` between `hundred` and `fifty` -> not really an error? 
```shell script
["hundred fifty", {"hundred and fifty": 104, "one hundred and": 1, "a hundred and": 1}]
```

## 2. "fine-tune" KenLM
1. train `enhanced KenLM` on __enhanced train-corpus__ = __librispeech-lm-data__ + __correction ngrams__ of erroneous phrases from [TEDLIUMv2 trainset](https://www.openslr.org/19/)
    -> should give `hundred fifty` slightly higher probability
2. Got WER of 0.289 vs. 0.284 ([see](learning_from_errors_kenlm.ipynb)) -> makes not really a difference!

## details
* __decoding__ is done with CTC prefix beam search + ngram LM implemented by [NeMo](https://github.com/NVIDIA/NeMo) and [OpenSeq2Seq](https://github.com/NVIDIA/OpenSeq2Seq/tree/master/decoders)
* __correction ngrams__: 
    * only 3-grams where taken -> ~1mio (800k unique)
    * added 10xtimes to __librispeech-lm-data__ -> [see](build_kenlm.py)
* impact on kenlm-arpa-file
```shell script
hyp: were learning about celoron armchairs and once we figure out
Ref: we're learning about cellular mechanics once we figure out
zcat ngrams.txt.gz | rg --color=always "cellular" | wc -l -> 134
zcat kenlm_cache_vanilla/lower.txt.gz | rg cellular | wc -l -> 324
zcat kenlm_cache_tedlium/lower.txt.gz | rg cellular | wc -l -> 1664
cat kenlm_vanilla/lm_filtered.arpa | rg "\tcellular\t" -> -5.7082434
cat kenlm_tedlium/lm_filtered.arpa | rg "\tcellular\t" -> -5.644795
```

## preparing data
* [processing TEDLIUMv2](https://github.com/dertilo/speech-recognition/blob/master/data_related/datasets/speech_corpora.py)
* [notebook](speech_data.ipynb) downloading + mp3-converting librispeech data on google colab -> takes ages but is for free
```shell script
found: /mydrive/data/asr_data/ENGLISH/train-clean-100.tar.gz no need to download
wrote /mydrive/data/asr_data/ENGLISH/train-clean-100_processed_mp3.tar.gz
downloading: http://www.openslr.org/resources/12/train-clean-360.tar.gz
wrote /mydrive/data/asr_data/ENGLISH/train-clean-360_processed_mp3.tar.gz
28539it [33:54, 14.03it/s]
104014it [2:13:54, 12.95it/s]
CPU times: user 5.85 s, sys: 1.07 s, total: 6.92 s
Wall time: 5h 31min 3s
```
