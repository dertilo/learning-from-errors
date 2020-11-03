# LFC 

## approaches
#### 1. "finetune" KenLM 
* nemo quartznet inference on librispeech wav-version and low quality mp3
* train KenLM on [normalized data](http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz); I suppose thats how LM models at [openslr](http://www.openslr.org/11/) where created
* train `enhanced KenLM` on "modified" train-corpus which constists of original `normalized data` and corrections of erroneous phrases from train-960
* erroneous phrases found by inference on low quality train-960 and alignment method (stolen from kaldi)
* evaluation: WER of ASR-system quartz+KenLM and quartz+enhanced-KenLM on: 
    1. dev-other wav -> hopefully WER of enhanced KenLM gets not worse!
    2. dev-other low quality -> hypothesis enhanced KenLM get lower WER then standard KenLM

* assuming shallow fusion of probabilities of acoustic model and language model
    + tweaking the language model (+ train data) in a way that it compensates for bias in the acoustic models


#### 2. use SOTA LM and finetune it
* needs beam-search that can handle "external" LM as scorer
* LM integration into ASR, shallow-fusion weights?
    * espnet config for librispeech proposes: `lm_weight: 0.6 ctc_weight: 0.4` -> !?
 
#### 3. "finetune" the acoustic model (quartznet) 
* fine-tune nemo quartznet
    * on (speech,text) dataset containing corrections

#### 4. finetune seq2seq model on correction pairs
* finetune BART?

## setup
1. install [nemo](https://github.com/NVIDIA/NeMo) 
```shell script
conda create -n lfc python=3.8 -y
sudo apt-get update && apt-get install -qq bc tree sox ffmpeg libsndfile1 libsox-fmt-mp3

# NeMo itself
conda activate lfc #or source activate on colab
git clone https://github.com/NVIDIA/NeMo -b main # when exactly did github rename master to main?!
pip -q install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html # caution! this depends on cuda version! (colab uses 10.1) default pytorch installation takes 10.2
cd NeMo/ && pip -q install -r requirements/requirements.txt
cd NeMo/ && pip -q install -e .[asr]

# stuff I like:
pip install wandb torchaudio
pip install "util@git+https://git@github.com/dertilo/util.git#egg=util"
```
2. install [ctc_decoders](https://github.com/NVIDIA/NeMo/blob/main/scripts/install_ctc_decoders.sh)
```shell script
sudo apt-get install swig
cd NeMo && scripts/install_ctc_decoders.sh
```
## preparing data

* [librispeech data](http://www.openslr.org/12/)
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

#### generate erroneous data
1. `nemo quartznet + kenlm` inference on librispeech
    * kenlm: 3 gram, minfreqs: [0,8,9]
```shell script
python3 kenlm_arpa.py --input_txt $HOME/data/asr_data/ENGLISH/librispeech-lm-norm.txt.gz --output_dir ./kenlm_3_089   --top_k 200000 --kenlm_bins $HOME/code/CPP/kenlm/build/bin/   --arpa_order 3 --max_arpa_memory "20%" --arpa_prune "0|8|9"
gzip -c kenlm_3_089/lm_filtered.arpa > ~/googledrive/data/asr_data/ENGLISH/kenlm_3_089.arpa.gz
```
2. calculate [erroneous ngrams](erroneous_ngrams.py)


