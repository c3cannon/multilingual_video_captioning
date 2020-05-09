#Dataloader
import json
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pathlib
from os import listdir
from os.path import isfile, join



class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode, language):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data
        
        self.language = language
        # load the json file which contains information about the dataset

        if self.language == 'english':
            self.captions = json.load(open(opt["eng_caption_json"]))
            info = json.load(open(opt["eng_info_json"]))
        elif self.language == 'chinese':
            self.captions = json.load(open(opt["chin_caption_json"]))
            info = json.load(open(opt["chin_info_json"]))      


        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))

        self.i3d_feats_dir = opt['i3d_feats_dir']
        print('load feats from %s' % (self.i3d_feats_dir))

        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

        self.filelist = list(info['videos'][self.mode])
        print(len(self.filelist))

        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        
        file = self.filelist[ix]

        i3d_feat = np.load(os.path.join(self.i3d_feats_dir, '{0}.npy'.format(file)))
        #i3d_feat = np.mean(i3d_feat, axis=0, keepdims=True)

        label = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)
        captions = self.captions[file]['final_captions']
        gts = np.zeros((len(captions), self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]

        cap_ix = random.randint(0, len(captions) - 1)
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1

        i3d_feat.resize(1, 32, 1024)

        data = {}
        data['i3d_feats'] = torch.from_numpy(i3d_feat).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['gts'] = torch.from_numpy(gts).long()
        data['video_ids'] = file

        return data

    def __len__(self):
        return len(self.filelist)