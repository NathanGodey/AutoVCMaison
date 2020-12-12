"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch
from torch_utils import device
import argparse

def make_metadata(dataset_dir = 'training_set'):
    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
    if torch.cuda.is_available():
        c_checkpoint = torch.load('3000000-BL.ckpt')
    else:
        c_checkpoint = torch.load('3000000-BL.ckpt', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)
    num_uttrs = 10
    len_crop = 128

    # Directory containing mel-spectrograms
    rootDir = dataset_dir + '/spmel'
    dirName, subdirList, _ = next(os.walk(rootDir))
    print('Found directory: %s' % dirName)

    print(subdirList)
    speakers = []
    for speaker in sorted(subdirList):
        print('Processing speaker: %s' % speaker)
        utterances = []
        utterances.append(speaker)
        fileList = []
        for root, _, files in os.walk(os.path.join(dirName,speaker)):
            for fileName in files:
                fileList.append(os.path.join(root,fileName))
        # make speaker embedding
        if len(fileList) < num_uttrs:
            print(f'Could not process speaker {speaker} : not enough files were found ({len(fileList)})')
            continue
        idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
        embs = []
        for i in range(num_uttrs):
            tmp = np.load(os.path.join(fileList[idx_uttrs[i]]))
            candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
            # choose another utterance if the current one is too short
            while tmp.shape[0] < len_crop:
                idx_alt = np.random.choice(candidates)
                tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
                candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
            left = np.random.randint(0, tmp.shape[0]-len_crop)
            melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).to(device)
            emb = C(melsp)
            embs.append(emb.detach().squeeze().cpu().numpy())
        utterances.append(np.mean(embs, axis=0))

        # create file list
        for fileName in sorted(fileList):
            fileName = fileName.replace('\\', '/')
            utterances.append('/'.join(fileName.split('/')[-2:]))
        speakers.append(utterances)

    with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
        pickle.dump(speakers, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset dir
    parser.add_argument('--dataset', type=str, default="voxceleb", help='dataset dir')
    config = parser.parse_args()
    make_metadata(config.dataset)
