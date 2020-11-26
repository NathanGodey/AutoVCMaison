import torch
import librosa
import pickle
from synthesis import build_model
from synthesis import wavegen
import soundfile as sf
from torch_utils import device

spect_vc = pickle.load(open('results.pkl', 'rb'))
model = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)
    sf.write(name+'.wav', waveform, 16000)
