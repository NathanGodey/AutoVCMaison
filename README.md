## <a href=https://github.com/NathanGodey/AutoVCMaison/blob/main/CycleAutoVC_REPORT.pdf>CYCLEAUTOVC: Cycle-Consistent Auto-Encoder For Few-Shot Voice Conversion</a>

This work is the main assignment for the CentraleSupelec course <i>Deep Learning</i> led by Valentin Petit and Maria Vakalopolou. You can find the report <a href=https://github.com/NathanGodey/AutoVCMaison/blob/main/CycleAutoVC_REPORT.pdf>HERE</a>.

### Audio Demo of AutoVC (from original authors)

The audio demo for AUTOVC can be found [here](https://auspicious3000.github.io/autovc-demo/)

### Dependencies
- Python 3
- Numpy
- PyTorch >= v0.4.1
- TensorFlow >= v1.3 (only for tensorboard)
- librosa
- tqdm
- wavenet_vocoder ```pip install wavenet_vocoder```
  for more information, please refer to https://github.com/r9y9/wavenet_vocoder

### Pre-trained models

| AUTOVC | Speaker Encoder | WaveNet Vocoder |
|----------------|----------------|----------------|
| [link](https://drive.google.com/file/d/1SZPPnWAgpGrh0gQ7bXQJXXjOntbh4hmz/view?usp=sharing)| [link](https://drive.google.com/file/d/1ORAeb4DlS_65WDkQN6LHx5dPyCM5PAVV/view?usp=sharing) | [link](https://drive.google.com/file/d/1Zksy0ndlDezo9wclQNZYkGi_6i7zi4nQ/view?usp=sharing) |


### 0.Voice Conversion
If you want to apply the style of speaker p228 to the file ```p225/p225_003.wav```, run :

```python converter.py --source='p225/p225_003.wav' --target='p228'```



### 2.Train model

We have included a small set of training audio files in the wav folder. However, the data is very small and is for code verification purpose only. Please prepare your own dataset for training.

1.Generate spectrogram data from the wav files: ```py .\make_spect.py --dataset='voxceleb'```

2.Generate training metadata, including the GE2E speaker embedding (please use one-hot embeddings if you are not doing zero-shot conversion): ```py .\make_metadata.py --dataset='voxceleb'```

3.Run the main training script: ```python main.py``` or ```python main_circular.py``` for CycleAutoVC. You can provide several parameters for the training in the bash command (learning rate, dataset, bottleneck dimension, ...). To display the list of parameters : ```python main(_circular).py -h```



