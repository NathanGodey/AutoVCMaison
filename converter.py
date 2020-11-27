import argparse
import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator
from torch_utils import device
import librosa
from synthesis import build_model
from synthesis import wavegen
import soundfile as sf
from torch_utils import device

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='autovc.ckpt')
parser.add_argument("--source")
parser.add_argument("--target")
parser.add_argument("--spmelFolder", default='./spmel')
parser.add_argument("--wavsFolder", default='./wavs')
parser.add_argument("--metadata", default='spmel/train.pkl')
parser.add_argument("--vocoder", default='checkpoint_step001000000_ema.pth')
parser.add_argument("--outputFolder", default='results')
args = parser.parse_args()

source_person = args.source.split('/')[0]
target_person = args.target.split('/')[0]

if not os.path.isdir(args.outputFolder):
    os.mkdir(args.outputFolder)

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

def get_embedding(metadata, speaker):
    for sbmt_i in metadata:
        if sbmt_i[0] == speaker:
            return torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
    raise Exception(f'Embedding was not found for speaker {speaker}.')
    #TODO : generate embedding from David's functions

def get_uttr_melspect(uttr_wav_path):
    uttr_spmel_path = os.path.join(args.spmelFolder,uttr_wav_path[:-4]+'.npy')
    mel_spect_exists = os.path.isfile(uttr_spmel_path)
    if mel_spect_exists:
        mlspect = np.load(uttr_spmel_path)
    else:
        #TODO : implement auto-convert
        raise Exception(f'The spectogram for {uttr_wav_path} does not exist, auto-convert is not supported yet.')
    return mlspect


with torch.no_grad():
    G = Generator(32,256,512,32).eval().to(device)
    g_checkpoint = torch.load(args.model, map_location=device)
    G.load_state_dict(g_checkpoint['model'])
    metadata = pickle.load(open(args.metadata, "rb"))
    spect_vc = []

    emb_org = get_embedding(metadata, source_person)
    emb_trg = get_embedding(metadata, target_person)

    source_path = os.path.join(args.wavsFolder,args.source)
    if os.path.isfile(source_path):
        X_orgs = [args.source]
    elif os.path.isdir(source_path):
        X_orgs = [os.path.join(args.source,file) for _,_,files in os.walk(source_path) for file in files]
    else:
        raise Exception(f'Wrong path: {source_path}')

    for x_org_source in X_orgs:
        source_file = '__'.join(x_org_source.split('/')[1:])
        x_org = get_uttr_melspect(x_org_source)
        x_org, len_pad = pad_seq(x_org)
        uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)

        _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
        spect_vc.append( ('{}_by_{}'.format(source_file[:-4], target_person), uttr_trg) )

    del G
    del g_checkpoint

    model = build_model().to(device)
    checkpoint = torch.load(args.vocoder)
    model.load_state_dict(checkpoint["state_dict"])

    for spect in spect_vc:
        name = spect[0]
        c = spect[1]
        print(name)
        waveform = wavegen(model, c=c)
        sf.write(f'{args.outputFolder}/{name}.wav', waveform, 16000)
