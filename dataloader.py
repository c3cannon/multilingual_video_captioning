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

    def __init__(self, opt, mode):
        super(VideoDataset, self).__init__()
        self.mode = "train"  # to load train/val/test data
        
        # load the json file which contains information about the dataset
        self.captions = json.load(open(opt["caption_json"]))
        info = json.load(open(opt["info_json"]))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))

        self.i3d_feats_dir = opt['i3d_feats_dir']
        print('load feats from %s' % (self.i3d_feats_dir))

        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

        #self.filelist = [f for f in listdir(self.i3d_feats_dir) if isfile(join(self.i3d_feats_dir, f))]
        self.filelist = list(self.captions.keys())
        print(len(self.filelist))

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