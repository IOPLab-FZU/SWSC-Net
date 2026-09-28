import torch
from torch.utils.data import Dataset
from preprocess_data import encode
import json
import os
import numpy as np
from imageio import imread
from random import *

class WHUCDCDataset(Dataset):

    def __init__(self, data_folder, list_path, split, token_folder=None, vocab_file=None, max_length=40, allow_unk=0, max_iters=None):
        self.mean=[123.6404, 118.4033, 108.3062]
        self.std=[49.7608, 48.1703, 51.4022]
        self.list_path = list_path
        self.split = split
        self.max_length = max_length

        assert self.split in {'train', 'val', 'test'}
        self.img_ids = [i_id.strip() for i_id in open(os.path.join(list_path + split + '.txt'))]
        if vocab_file is not None:
            with open(os.path.join(list_path + vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)
            self.allow_unk = allow_unk
        if max_iters is not None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]
        self.files = []
        if split == 'train':
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split + '/A/' + name.split('-')[0])
                img_fileB = img_fileA.replace('/A/', '/B/')
                token_id = name.split('-')[-1]
                token_file = os.path.join(token_folder + name.split('.')[0] + '.txt') if token_folder is not None else None
                self.files.append({
                    'imgA': img_fileA,
                    'imgB': img_fileB,
                    'token': token_file,
                    'token_id': token_id,
                    'name': name.split('-')[0]
                })
        else:
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + '/' + split + '/A/' + name)
                img_fileB = img_fileA.replace('/A/', '/B/')
                token_file = os.path.join(token_folder + name.split('.')[0] + '.txt') if token_folder is not None else None
                self.files.append({
                    'imgA': img_fileA,
                    'imgB': img_fileB,
                    'token': token_file,
                    'token_id': None,
                    'name': name
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles['name']
        imgA = imread(datafiles['imgA'])
        imgB = imread(datafiles['imgB'])
        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)
        imgA = np.moveaxis(imgA, -1, 0)
        imgB = np.moveaxis(imgB, -1, 0)

        for i in range(len(self.mean)):
            imgA[i, :, :] -= self.mean[i]
            imgA[i, :, :] /= self.std[i]
            imgB[i, :, :] -= self.mean[i]
            imgB[i, :, :] /= self.std[i]

        if datafiles['token'] is not None:
            caption = open(datafiles['token']).read()
            caption_list = json.loads(caption)

            token_all = np.zeros((len(caption_list), self.max_length), dtype=int)
            token_all_len = np.zeros((len(caption_list), 1), dtype=int)
            for j, tokens in enumerate(caption_list):
                tokens_encode = encode(tokens, self.word_vocab, allow_unk=self.allow_unk == 1)
                token_all[j, :len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)
            if datafiles['token_id'] is not None:
                idx = int(datafiles['token_id'])
                token = token_all[idx]
                token_len = token_all_len[idx].item()
            else:
                j = randint(0, len(caption_list) - 1)
                token = token_all[j]
                token_len = token_all_len[j].item()
        else:
            token_all = np.zeros(1, dtype=int)
            token = np.zeros(1, dtype=int)
            token_len = np.zeros(1, dtype=int)
            token_all_len = np.zeros(1, dtype=int)

        return imgA.copy(), imgB.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len), name
