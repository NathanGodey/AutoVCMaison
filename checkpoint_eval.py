from converter import converter
import argparse
import os
import torch
from torch_utils import device

def checkpoint_eval(dataset,
    vocoder = 'checkpoint_step001000000_ema.pth', outputFolder ='results', checkpoints_dir='trained_models'):
    spmelFolder = os.path.join(dataset,'spmel')
    wavsFolder = os.path.join(dataset,'wavs')
    print("Doing conversion for each checkpoints...")
    target = os.listdir(wavsFolder)[0]
    source_id = os.listdir(wavsFolder)[1]
    source = ''
    for dirName,_,files in os.walk(os.path.join(wavsFolder,source_id)):
        if len(files)!=0:
            dirName = dirName.replace('\\', '/')
            dirName = '/'.join(dirName.split('/')[2:])
            source = os.path.join(dirName,files[0])
            break
    _, _, files = next(os.walk(checkpoints_dir))
    for checkpoint in files:
        if not os.path.exists(os.path.join(checkpoints_dir,checkpoint+'_sound')):
            os.makedirs(os.path.join(checkpoints_dir,checkpoint+'_sound'))
        print('Found checkpoint: ',checkpoint)
        converter(os.path.join(checkpoints_dir,checkpoint), source, target, spmelFolder, wavsFolder, os.path.join(spmelFolder,'train.pkl'), outputFolder=os.path.join(checkpoints_dir,checkpoint+'_sound'))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="voxceleb", help='dataset dir')
    args = parser.parse_args() 

    checkpoint_eval(args.dataset)