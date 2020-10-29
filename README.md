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

#### 2. use SOTA LM and finetune it
 
#### 3. "finetune" the acoustic model (quartznet) 


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

## get data
* with my own script !


