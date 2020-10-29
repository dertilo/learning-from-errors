import nemo.collections.asr as nemo_asr
import numpy as np
import os

if __name__ == '__main__':
    # nemo_asr.models.EncDecCTCModel.list_available_models()
    # asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name='QuartzNet15x5Base-En', strict=False) # what is
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet5x5LS-En")

    # softmax implementation in NumPy
    def softmax(logits):
        e = np.exp(logits - np.max(logits))
        return e / e.sum(axis=-1).reshape([logits.shape[0], 1])


    data_path = f"{os.environ['HOME']}/data/asr_data/ENGLISH/LibriSpeech/dev-clean/2277/149897"
    files = [f"{data_path}/2277-149897-0024.flac"]

    transcript = asr_model.transcribe(paths2audio_files=files)[0]
    print(f'Transcript: "{transcript}"')

    logits = asr_model.transcribe(files, logprobs=True)[0].cpu().numpy()
    probs = softmax(logits)
    print(probs)