"""
Author: Omkar Damle
Date: 28th Oct 2018

Dataloader class
Points to note:
1. The annotations are returned as an array of indices according to the vocabulary dictionary
2. Each annotation starts with the start symbol and ends with the end symbol. There can be padding after the end symbol in order
to make a batch
3. Each image feature has the shape - (36,2048)

reference: 
https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/dataset.py
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
"""


import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from data_helpers.vocab import Vocabulary
from pycocotools.coco import COCO

# import h5py
import argparse
import pickle

def indexto1hot(vocab_len, index):
    #print("index type: ")
    if isinstance(index,int) == False:
        n = len(index)
        one_hot = np.zeros([n, vocab_len])
        for i in range(n):
            one_hot[i,index[i]] = 1
        return one_hot
    else:
        one_hot = np.zeros([vocab_len])
        one_hot[index] = 1
        return one_hot

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - feature: torch tensor of shape (36,2048).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        features: torch tensor of shape (batch_size, 36, 2048).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    #print("Length of list: " + str(len(data)))

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[2]), reverse=True)
    image_ids, features, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    features = torch.stack(features, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]

    #vocab_len = len(captions[0][0])
    
    #print("Vocab len: " + str(vocab_len) + "\n")
    #print("Type: " + type(captions[0]))
    #print("\n")

    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return image_ids, features, targets, lengths

