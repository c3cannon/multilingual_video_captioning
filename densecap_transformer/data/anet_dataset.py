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
from collections import defaultdict
import math
import multiprocessing
import pickle
from random import shuffle, choice

import torch
import torchtext
from torch.utils.data import Dataset

from data.utils import segment_iou

def process_data(dataset_file, split):
    # process train or validation dataset

    sentences = []
    nvideos = 0

    with open(dataset_file, "r") as data_file:
        data = json.load(data_file)

    for row in data:
        en_anns = row['enCap']
        ch_anns = row["chCap"]
        nvideos += 1
        for ind, ann in enumerate(en_anns):
            ann = ann.strip()
            # if split == "training":
            #     train_sentences.append(ann['sentence'])
            sentences.append(ann)
        row["subset"] = split

    print("LEN SENTENCES", len(sentences))

    return sentences, nvideos, data



def get_vocab_and_sentences(train_dataset_file, val_dataset_file, max_length=20):
    # build vocab and tokenized sentences
    text_proc = torchtext.data.Field(sequential=True, init_token='<init>',
                                eos_token='<eos>', tokenize='spacy',
                                lower=True, batch_first=True,
                                fix_length=max_length)

    print("train data file:", train_dataset_file)
    # print("val data file:", val_dataset_file)

    train_sentences, ntrain_videos, data_all = process_data(train_dataset_file, "train")
    # val_sentences, nval_videos, val_data = process_data(val_dataset_file, "validation")
    # data_all.extend(val_data)

    nsentence = {}
    nsentence["training"] = ntrain_videos
    # nsentence["validation"] = nval_videos

    sentences_proc = list(map(text_proc.preprocess, train_sentences)) # build vocab on train only
    # sentences_proc = list(map(text_proc.preprocess, train_sentences + val_sentences)) # build vocab on train and val
    text_proc.build_vocab(sentences_proc)#, min_freq=5)
    print('# of words in the vocab: {}'.format(len(text_proc.vocab)))
    # print(
    #     '# of sentences in training: {}, # of sentences in validation: {}'.format(
    #         nsentence['training'], nsentence['validation']
    #     ))
    print(
        '# of sentences in training: {}'.format(
            nsentence['training']
        ))
    print('# of training videos: {}'.format(ntrain_videos))

    return text_proc, data_all

# dataloader for training
class ANetDataset(Dataset):
    def __init__(self, vid_path, split, slide_window_size,
                 dur_file, kernel_list, text_proc, raw_data,
                 stride_factor, dataset, save_samplelist=False,
                 load_samplelist=False, sample_listpath=None):
        super(ANetDataset, self).__init__()

        self.slide_window_size = slide_window_size

        if not load_samplelist:
            self.sample_list = []  # list of list for data samples

            ch_train_sentences = []
            en_train_sentences = []
            for val in raw_data:
                en_annotations = val["enCap"]
                ch_annotations = val["chCap"]
                vid = val["videoID"]
                if val['subset'] == "train" and os.path.isfile(os.path.join(vid_path, vid + '.npy')):
                    for ann in en_annotations:
                        ann = ann.strip()
                        en_train_sentences.append(ann)
                    for ann in ch_annotations:
                        ann = ann.strip()
                        ch_train_sentences.append(ann)

            en_train_sentences = list(map(text_proc.preprocess, en_train_sentences))
            print("len EN_TRAIN_SENTENCES:", len(en_train_sentences))
            en_sentence_idx = text_proc.numericalize(text_proc.pad(en_train_sentences))#,
                                                       # device=-1)  # put in memory
            print("EN_SENTENCE_IDX:", en_sentence_idx[0])
            print("len EN_SENTENCE_IDX:", len(en_sentence_idx))
            if en_sentence_idx.size(0) != len(en_train_sentences):
                raise Exception("Error in numericalize English sentences")
            
            ch_train_sentences = list(map(text_proc.preprocess, ch_train_sentences))
            ch_sentence_idx = text_proc.numericalize(text_proc.pad(ch_train_sentences))
            if ch_sentence_idx.size(0) != len(ch_train_sentences):
                raise Exception("Error in numericalize Chinese sentences")

            # load annotation per video and construct training set
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                results = [None]*(len(raw_data)*10) # multiply by 10 b/c 10 English captions/video
                vid_idx = 0
                for i,row in enumerate(raw_data):
                    en_annotations = val["enCap"]
                    vid = val["videoID"]
                    if val["subset"] == "train" and os.path.isfile(os.path.join(vid_path, vid + '.npy')):
                        for j,ann in enumerate(en_annotations):
                            # results[vid_idx] = pool.apply_async(_get_pos_neg,
                            #          (vid_path, raw_data, vid, ann))
                            results.append((os.path.join(vid_path, vid), ann, en_sentence_idx[vid_idx]))
                            vid_idx += 1
                # results = results[:vid_idx]
                # for i, r in enumerate(results):
                #     results[i] = r.get()

            for r in results:
                if r is not None:
                    video_prefix, sent, sent_idx = r
                    self.sample_list.append((video_prefix, sent, sent_idx))

            print('total number of {} videos: {}'.format(split, len(raw_data)))
            print('total number of {} samples (unique pairs): {}'.format(
                split, len(self.sample_list)))
            print('total number of English annotations: {}'.format(len(en_train_sentences)))
            print('total number of Chinese annotations: {}'.format(len(ch_train_sentences)))

            if save_samplelist:
                with open(sample_listpath, 'wb') as f:
                    pickle.dump(self.sample_list, f)
        else:
            with open(sample_listpath, 'rb') as f:
                self.sample_list = pickle.load(f)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        video_prefix, sentence, sentence_idx = self.sample_list[index]
        img_feat = torch.from_numpy(
            np.load(video_prefix + '.npy')).squeeze(0).float()
        # torch.cat((resnet_feat, bn_feat), dim=1,
        #           out=img_feat[:min(total_frame, self.slide_window_size)])

        return (sentence, sentence_idx, img_feat)


# def _get_pos_neg(vid_path, raw_data, vid, pos_ann):
#                  # pos_thresh, neg_thresh):
#     # Naive neg sampling for now (random sampling)
#     # In the future, could use either the captions or the video features
#     # to choose better negatives
#     if os.path.isfile(os.path.join(vid_path, vid + '.npy')):
#         print('video: {}'.format(vid))

#         video_prefix = os.path.join(vid_path, vid)

#         # choice(raw_data) gets a list of captions for one image
#         # choice(choice(raw_data)["enCap"]) gets a single caption
#         neg_sample = choice(raw_data)
#         while neg_sample["videoID"] == vid:
#             neg_sample = choice(raw_data)

#         # randomly sample an english caption for now
#         neg_ann = choice(neg_sample["enCap"])

#         return video_prefix, neg_ann, pos_ann
#     else:
#         return None


def anet_collate_fn(batch_lst):
    # each item of batch_lst is a tuple (sentence, vid_features)

    sample_each = 10  # TODO, hard coded
    sentence, sentence_idx, img_feat = batch_lst[0]

    batch_size = len(batch_lst)

    sentence_batch = torch.LongTensor(np.ones((batch_size, sentence_idx.size(0)), dtype='int64'))
    img_batch = torch.FloatTensor(np.zeros((batch_size,
                                            img_feat.size(0),
                                            img_feat.size(1))))
    # tempo_seg_pos = torch.FloatTensor(np.zeros((batch_size, sample_each, 4)))
    # tempo_seg_neg = torch.FloatTensor(np.zeros((batch_size, sample_each, 2)))

    for batch_idx in range(batch_size):
        sentence, sentence_idx, img_feat = batch_lst[batch_idx]

        print("len sentence:", len(sentence))

        print("sentence:", sentence)
        print("batch size:", batch_size)
        print("img batch shape:", img_batch.size()) # 32 32 1024
        print("img feat shape:", img_feat.size()) # 1 32 1024
        print("batch idx:", batch_idx)
        print("sentence idx shape:", sentence_idx.size())
        print("sentence batch shape:", sentence_batch.size())
        print("sentence_idx.data:", sentence_idx.data)

        img_batch[batch_idx,:] = img_feat

        sentence_batch[batch_idx] = sentence_idx

        # # sample positive anchors
        # perm_idx = torch.randperm(len(pos_seg))
        # if len(pos_seg) >= sample_each:
        #     tempo_seg_pos[batch_idx,:,:] = pos_seg_tensor[perm_idx[:sample_each]]
        # else:
        #     tempo_seg_pos[batch_idx,:len(pos_seg),:] = pos_seg_tensor
        #     idx = torch.multinomial(torch.ones(len(pos_seg)), sample_each-len(pos_seg), True)
        #     tempo_seg_pos[batch_idx,len(pos_seg):,:] = pos_seg_tensor[idx]

        # # sample negative anchors
        # neg_seg_tensor = torch.FloatTensor(neg_seg)
        # perm_idx = torch.randperm(len(neg_seg))
        # if len(neg_seg) >= sample_each:
        #     tempo_seg_neg[batch_idx, :, :] = neg_seg_tensor[perm_idx[:sample_each]]
        # else:
        #     tempo_seg_neg[batch_idx, :len(neg_seg), :] = neg_seg_tensor
        #     idx = torch.multinomial(torch.ones(len(neg_seg)),
        #                             sample_each - len(neg_seg),True)
        #     tempo_seg_neg[batch_idx, len(neg_seg):, :] = neg_seg_tensor[idx]

    # return (img_batch, tempo_seg_pos, tempo_seg_neg, sentence_batch)
    return (img_batch, sentence_batch)
