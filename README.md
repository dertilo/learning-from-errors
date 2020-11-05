# LFC 
**cryptic naming is on purpose**

## "finetune" KenLM 
* based on CTC prefix beam search + ngram LM implemented by [NeMo](https://github.com/NVIDIA/NeMo) and [OpenSeq2Seq](https://github.com/NVIDIA/OpenSeq2Seq/tree/master/decoders)
* assuming shallow fusion of probabilities of acoustic model and language model
    + tweaking the language model (+ train data) in a way that it compensates for bias in the acoustic models

1. train KenLM on [__librispeech-lm-data__](http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz); I suppose thats how LM models at [openslr](http://www.openslr.org/11/) where created
    -> receive __librispeech-KenLM__
2. train `enhanced KenLM` on __enhanced train-corpus__ which consists of original __librispeech-lm-data__ and __correction ngrams__ of erroneous phrases from [TEDLIUMv2 trainset](https://www.openslr.org/19/)
* __correction ngrams__:
    1. inference of [NeMo QuartzNet](https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels) + __librispeech-KenLM__ on [TEDLIUMv2 trainset](https://www.openslr.org/19/) 
    2. [alignment method](alignment.py) (stolen from kaldi) to find erroneous ngrams
* [evaluation via WER](learning_from_errors_kenlm.ipynb) 

## preparing data
* [notebook](speech_data.ipynb) dowloading + mp3-converting librispeech data on google colab -> takes ages but is for free
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
* TEDLIUMv2 [see](https://github.com/dertilo/speech-recognition/blob/master/data_related/datasets/speech_corpora.py)

#### generate erroneous data
1. build __librispeech-KenLM__
```shell script
python3 kenlm_arpa.py --input_txt $HOME/data/asr_data/ENGLISH/librispeech-lm-norm.txt.gz --output_dir ./kenlm_3_089   --top_k 200000 --kenlm_bins $HOME/code/CPP/kenlm/build/bin/   --arpa_order 3 --max_arpa_memory "20%" --arpa_prune "0|8|9"
gzip -c kenlm_3_089/lm_filtered.arpa > ~/googledrive/data/asr_data/ENGLISH/kenlm_3_089.arpa.gz
```
2. use __librispeech-KenLM__ to transcribe [TEDLIUMv2-trainset](http://www.openslr.org/19/)
    * trainset has 90979 samples

3. calculate [erroneous ngrams](erroneous_ngrams.py)


