"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import numpy as np
import csv
import math
import multiprocessing
import pickle
from random import shuffle, choice
import string

import torch
import torchtext
from torch.utils.data import Dataset

import torch.nn.functional as F

TABLE = str.maketrans('', '', string.punctuation)

def process_data(dataset_file):

    print("DATASET_FILE:", dataset_file)
    # process train or validation dataset

    sentences = []
    seen = set()
    nvideos = 0

    with open(dataset_file, "r") as data_file:
        train_data = json.load(data_file)

    for row in train_data:
        en_anns = row['enCap']
        ch_anns = row["chCap"]
        nvideos += 1
        for ind, (en_ann, ch_ann) in enumerate(zip(en_anns, ch_anns)):
            en_ann = " ".join([w.translate(TABLE) for w in en_ann.strip().lower()])
            ch_ann = " ".join([w.translate(TABLE) for w in ch_ann.strip().lower()])
            # if split == "training":
            #     train_sentences.append(ann['sentence'])
            sentences.append(en_ann)
            sentences.append(ch_ann)
            seen.add(en_ann)
        row["subset"] = "train"

    with open(dataset_file.replace("training", "validation"), "r") as data_file:
        val_data = json.load(data_file)

    for row in val_data:
        en_anns = row['enCap']
        ch_anns = row["chCap"]
        nvideos += 1
        for ind, (en_ann, ch_ann) in enumerate(zip(en_anns, ch_anns)):
            en_ann = " ".join([w.translate(TABLE) for w in en_ann.strip().lower()])
            ch_ann = " ".join([w.translate(TABLE) for w in ch_ann.strip().lower()])
            # if split == "training":
            #     train_sentences.append(ann['sentence'])
            if en_ann not in seen:
                sentences.append(en_ann)
                sentences.append(ch_ann)
                seen.add(en_ann)
        row["subset"] = "validation" 

    return sentences, nvideos, train_data, val_data

def get_vocab_and_sentences(dataset_file, verbose=True):
    # build vocab and tokenized sentences
    text_proc = torchtext.data.Field(sequential=True, init_token='<sos>',
                                eos_token='<eos>',
                                lower=True, batch_first=True)

    sentences, nvideos, train_data, val_data = process_data(dataset_file)

    sentences_proc = list(map(text_proc.preprocess, sentences)) # build vocab on train only
    text_proc.build_vocab(sentences_proc)#, min_freq=5)
    
    if verbose:
        print('# of words in the train/val vocab: {}'.format(len(text_proc.vocab)))
        print('# of train/val videos: {}'.format(nvideos))

    return text_proc, train_data, val_data

# dataloader for training
class ANetDataset(Dataset):
    def __init__(self, vid_path, split,
                 text_proc, raw_data,
                 language, dset="train",
                 save_samplelist=False,
                 load_samplelist=False,
                 sample_listpath=None,
                 verbose=True):
        super(ANetDataset, self).__init__()

        if language != "en" and language != "ch":
            raise Exception("Error in language: {} not recognized".format(language))

        if not load_samplelist:
            # self.en_sample_list = []  # list of list for data samples
            # self.ch_sample_list = []  # list of list for data samples
            self.sample_list = []

            train_sentences = []
            for val in raw_data:
                annotations = val[language + "Cap"]
                vid = val["videoID"]
                if val['subset'] == dset and os.path.isfile(os.path.join(vid_path, vid + '.npy')):
                    for ann in annotations:
                        ann = " ".join([w.translate(TABLE) for w in ann.strip().lower().split()])
                        train_sentences.append(ann)

            train_sentences = list(map(text_proc.preprocess, train_sentences))

            sentence_idx = text_proc.numericalize(text_proc.pad(train_sentences))#,
                                                       # device=-1)  # put in memory

            if sentence_idx.size(0) != len(train_sentences):
                raise Exception("Error in numericalize {} sentences".format(language))

            # load annotation per video and construct training set
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                results = [None]*(len(raw_data)*10) # multiply by 10 b/c 10 captions/video per language
                sen_idx = 0
                for i,row in enumerate(raw_data):
                    annotations = row[language + "Cap"]
                    vid = row["videoID"]
                    if row["subset"] == dset and os.path.isfile(os.path.join(vid_path, vid + '.npy')):
                        for j, ann in enumerate(annotations):
                            ann = " ".join([w.translate(TABLE) for w in ann.strip().lower().split()])
                            results.append((os.path.join(vid_path, vid),
                                ann, sentence_idx[sen_idx]))
                            sen_idx += 1

            for i,r in enumerate(results):
                if r is not None:
                    video_prefix, sent, sent_idx = r
                    self.sample_list.append((video_prefix, sent, sent_idx))

            if verbose:
                print('total number of {} {} videos: {}'.format(split, dset, len(raw_data)))
                print('total number of {} {} samples (unique pairs): {}'.format(
                    language, split, len(self.sample_list)))
                print('total number of {} annotations: {}'.format(language,
                    len(train_sentences)))

            if save_samplelist:
                print("SAVING SAMPLE LIST TO PICKLE")
                with open(sample_listpath + dset, 'wb') as f:
                    pickle.dump(self.sample_list, f)
        else:
            print("LOADING SAMPLE LIST FROM PICKLE")
            with open(sample_listpath + dset, 'rb') as f:
                self.sample_list = pickle.load(f)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        video_prefix, sentence, sentence_idx = self.sample_list[index]
        img_feat = torch.from_numpy(
            np.load(video_prefix + '.npy')).squeeze(0).float()
        if img_feat.size()[0] != 32:
            odd_size = img_feat.size()[0]
            img_feat = F.pad(img_feat, pad=(0,0,0,32-odd_size), mode="constant", value=0)

        return (sentence, sentence_idx, img_feat, video_prefix)

def anet_collate_fn(batch_lst):
    # each item of batch_lst is a tuple (sentence, vid_features)

    sentence, sentence_idx, img_feat, video_prefix = batch_lst[0]

    batch_size = len(batch_lst)

    sentence_batch = torch.LongTensor(np.ones((batch_size, sentence_idx.size(0)), dtype='int64'))
    img_batch = torch.FloatTensor(np.zeros((batch_size,
                                            img_feat.size(0),
                                            img_feat.size(1))))
    video_prefixes = [None]*batch_size
    sentences = [None]*batch_size

    for batch_idx in range(batch_size):
        sentence, sentence_idx, img_feat, video_prefix = batch_lst[batch_idx]

        img_batch[batch_idx,:] = img_feat
        sentence_batch[batch_idx] = sentence_idx
        video_prefixes[batch_idx] = video_prefix
        sentences[batch_idx] = sentence

    lengths = [len(sent.split()) + 2 for sent in sentences]

    targets = torch.zeros(len(sentence_batch), max(lengths)).long()
    for i, sent in enumerate(sentence_batch):
        end = lengths[i]
        targets[i, :end] = sent[:end]

    # print("sentence batch shape: {}, targets shape: {}".format(sentence_batch.shape, targets.shape))

    return (img_batch, sentence_batch, sentences, video_prefixes, lengths)
