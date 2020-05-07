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
import re

import torch
import torchtext
from torch.utils.data import Dataset
import torch.nn.functional as F

import jieba

def process_ch(ch_ann):
    ch_ann = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", ch_ann)
    ch_ann = re.sub("[【】╮╯▽╰╭★→「」]+","",ch_ann)
    ch_ann = re.sub("！，❤。～《》：（）【】「」？”“；：、","",ch_ann)
    ch_ann = " ".join(jieba.cut(ch_ann.translate(str.maketrans('', '', string.punctuation)),
     cut_all=False))
    return ch_ann


def process_data(dataset_file, language):

    print("{} DATASET_FILE: {}".format(language, dataset_file))
    # process train or validation dataset

    sentences = []
    seen = set()
    nvideos = 0

    process_ann = lambda x: x.strip()
    if language == "ch":
        process_ann = process_ch
    else:
        process_ann = lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower()

    with open(dataset_file, "r") as data_file:
        train_data = json.load(data_file)

    for row in train_data:
        anns = row[language + 'Cap']
        nvideos += 1
        for ind, ann in enumerate(anns):
            ann = process_ann(ann)
            # if split == "training":
            #     train_sentences.append(ann['sentence'])
            sentences.append(ann)
            seen.add(ann)
        row["subset"] = "train"

    with open(dataset_file.replace("train", "val"), "r") as data_file:
        val_data = json.load(data_file)

    for row in val_data:
        anns = row[language + 'Cap']
        nvideos += 1
        for ind, ann in enumerate(anns):
            ann = process_ann(ann)
            # if split == "training":
            #     train_sentences.append(ann['sentence'])
            if ann not in seen:
                sentences.append(ann)
                seen.add(ann)
        row["subset"] = "validation" 

    null_sent = 0
    for sent in sentences:
        if sent is None:
            null_sent += 1

    return sentences, nvideos, train_data, val_data

def get_vocab_and_sentences(dataset_file, language, verbose=True):
    # build vocab and tokenized sentences
    text_proc = torchtext.data.Field(sequential=True, init_token='<sos>',
                                eos_token='<eos>',
                                lower=True, batch_first=True)

    sentences, nvideos, train_data, val_data = process_data(dataset_file, language)

    sentences_proc = list(map(text_proc.preprocess, sentences)) # build vocab on train only
    text_proc.build_vocab(sentences_proc)#, min_freq=5)

    if verbose:
        print('# of words in the {} vocab: {}'.format(dataset_file,
            len(text_proc.vocab)))
        print('# of {} videos: {}'.format(dataset_file, nvideos))

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
        print("language: {}, split: {}, vocab size: {}".format(language, split, len(text_proc.vocab)))

        self.process_ann = lambda x: x.strip()
        if language == "en":
            self.process_ann = lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower()
        else:
            self.process_ann = process_ch

        if not load_samplelist:
            # self.en_sample_list = []  # list of list for data samples
            # self.ch_sample_list = []  # list of list for data samples
            self.sample_list = []

            train_sentences = []
            for val in raw_data:
                annotations = val[language + "Cap"]
                vid = val["videoID"]
                if val['subset'] == dset and os.path.isfile(os.path.join(split, vid + '.npy')):
                    for ann in annotations:
                        ann = self.process_ann(ann)
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
                    if row["subset"] == dset and os.path.isfile(os.path.join(split, vid + '.npy')):
                        for j, ann in enumerate(annotations):
                            ann = self.process_ann(ann)
                            results.append((os.path.join(split, vid),
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

    return (img_batch, sentence_batch, sentences, video_prefixes, lengths)
