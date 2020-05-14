"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# general packages
import os
import errno
import argparse
import numpy as np
import random
import time
import yaml

# torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.distributed as dist
import torch.utils.data.distributed

# misc
from data.anet_dataset import ANetDataset, anet_collate_fn, get_vocab_and_sentences
from model.encoder_lstm import EncoderRNN, DecoderRNN, S2VTAttModel
from annoy import AnnoyIndex

from data.utils import update_values

import logging

parser = argparse.ArgumentParser()

# Data input settings
parser.add_argument('--cfgs_file', default='cfgs/vatex.yml', type=str, help='dataset specific settings')
parser.add_argument('--train_dataset_file', default='', type=str)
parser.add_argument('--val_dataset_file', default='', type=str)
parser.add_argument('--feature_root', default='', type=str, help='the feature root')
parser.add_argument('--train_data_folder', default=['training'], type=str, nargs='+', help='training data folder')
parser.add_argument('--val_data_folder', default=['validation'], help='validation data folder')
parser.add_argument('--save_train_samplelist', action='store_true')
parser.add_argument('--load_train_samplelist', action='store_true')
parser.add_argument('--train_samplelist_path', type=str, default='/z/home/luozhou/subsystem/densecap_vid/train_samplelist.pkl')
parser.add_argument('--save_valid_samplelist', action='store_true')
parser.add_argument('--load_valid_samplelist', action='store_true')
parser.add_argument('--valid_samplelist_path', type=str, default='/z/home/luozhou/subsystem/densecap_vid/valid_samplelist.pkl')
parser.add_argument('--start_from', default='', help='path to a model checkpoint to initialize model weights from. Empty = dont')
parser.add_argument('--load_nn', default='', type=str)
parser.add_argument('--save_nn', default='', type=str)
parser.add_argument('--num_trees', default=500, type=int)
parser.add_argument('--num_neighbors', default=5, type=int)
parser.add_argument('--max_sentence_len', default=20, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--mode', default='train', type=str, help='possible modes are \'train\' and \'inference\'')

# Model settings: General
parser.add_argument('--d_model', default=1024, type=int, help='size of the rnn in number of hidden nodes in each layer')
parser.add_argument('--d_hidden', default=512, type=int)
parser.add_argument('--d_word_emb', default=1000, type=int)
parser.add_argument('--attn_dropout', default=0.2, type=float)
parser.add_argument('--vis_emb_dropout', default=0.1, type=float)
parser.add_argument('--cap_dropout', default=0.2, type=float)
parser.add_argument('--image_feat_size', default=1024, type=int, help='the encoding size of the image feature')
parser.add_argument('--n_layers', default=2, type=int, help='number of layers in the sequence model')
parser.add_argument('--train_sample', default=20, type=int, help='total number of positive+negative training samples (2*U)')
parser.add_argument('--sample_prob', default=0, type=float, help='probability for use model samples during training')

# Optimization: General
parser.add_argument('--max_epochs', default=20, type=int, help='max number of epochs to run for')
parser.add_argument('--batch_size', default=32, type=int, help='what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
parser.add_argument('--valid_batch_size', default=64, type=int)
parser.add_argument('--sent_weight', default=0.25, type=float)
parser.add_argument('--scst_weight', default=0.0, type=float)

# Optimization
parser.add_argument('--optim',default='sgd', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
parser.add_argument('--teacher_forcing', type=float, default=0, help='teacher forcing ratio')
parser.add_argument('--alpha', default=0.95, type=float, help='alpha for adagrad/rmsprop/momentum/adam')
parser.add_argument('--beta', default=0.999, type=float, help='beta used for adam')
parser.add_argument('--epsilon', default=1e-8, help='epsilon that goes into denominator for smoothing')
parser.add_argument('--patience_epoch', default=1, type=int, help='Epoch to wait to determine a pateau')
parser.add_argument('--reduce_factor', default=0.5, type=float, help='Factor of learning rate reduction')

# Data parallel
parser.add_argument('--dist_url', default='file:///home/luozhou/nonexistent_file', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

# Evaluation/Checkpointing
parser.add_argument('--save_checkpoint_every', default=1, type=int, help='how many epochs to save a model checkpoint?')
parser.add_argument('--checkpoint_path', default='./checkpoint', help='folder to save checkpoints into (empty = this folder)')
parser.add_argument('--losses_log_every', default=1, type=int, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
parser.add_argument('--seed', default=213, type=int, help='random number generator seed to use')

# Logging
parser.add_argument('--logfile', default="densecap.log")


parser.set_defaults(cuda=False, save_train_samplelist=False,
                    load_train_samplelist=False,
                    save_valid_samplelist=False,
                    load_valid_samplelist=False,
                    gated_mask=False)

args = parser.parse_args()

with open(args.cfgs_file, 'r') as handle:
    options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
update_values(options_yaml, vars(args))

# print("PRINTING ARGS")
# print(args)

logging.basicConfig(filename=args.logfile, level=logging.INFO)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


def get_dataset(args):
    # process text
    en_text_proc, en_train_raw_data, en_val_raw_data = get_vocab_and_sentences(args.train_dataset_file, 
        "en", verbose=False)
    ch_text_proc, ch_train_raw_data, ch_val_raw_data = get_vocab_and_sentences(args.train_dataset_file, 
        "ch", verbose=False)

    # Create the dataset and data loader instance
    en_train_dataset = ANetDataset(args.feature_root,
                                args.train_data_folder,
                                en_text_proc, en_train_raw_data,
                                language="en",
                                save_samplelist=args.save_train_samplelist,
                                load_samplelist=args.load_train_samplelist,
                                sample_listpath=args.train_samplelist_path,
                                verbose=False)
    ch_train_dataset = ANetDataset(args.feature_root,
                                args.train_data_folder,
                                ch_text_proc, ch_train_raw_data,
                                language="ch",
                                save_samplelist=args.save_train_samplelist,
                                load_samplelist=args.load_train_samplelist,
                                sample_listpath=args.train_samplelist_path,
                                verbose=False)

    print("size of English train dataset:", len(en_train_dataset))
    print("size of Chinese train dataset:", len(ch_train_dataset))

    # dist parallel, optional
    args.distributed = args.world_size > 1
    if args.distributed and torch.cuda.is_available():
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    en_train_loader = DataLoader(en_train_dataset,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers,
                              collate_fn=anet_collate_fn)

    ch_train_loader = DataLoader(ch_train_dataset,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers,
                              collate_fn=anet_collate_fn)

    en_valid_dataset = ANetDataset(args.feature_root,
                                args.val_data_folder,
                                en_text_proc, en_val_raw_data,
                                language="en",
                                dset="validation",
                                save_samplelist=args.save_valid_samplelist,
                                load_samplelist=args.load_valid_samplelist,
                                sample_listpath=args.valid_samplelist_path,
                                verbose=False)

    en_valid_loader = DataLoader(en_valid_dataset,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=anet_collate_fn)
   
    ch_valid_dataset = ANetDataset(args.feature_root,
                                args.val_data_folder,
                                ch_text_proc, ch_val_raw_data,
                                language="ch",
                                dset="validation",
                                save_samplelist=args.save_valid_samplelist,
                                load_samplelist=args.load_valid_samplelist,
                                sample_listpath=args.valid_samplelist_path,
                                verbose=False)

    ch_valid_loader = DataLoader(ch_valid_dataset,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=anet_collate_fn)

    print("size of English val dataset:", len(en_valid_dataset))
    print("size of Chinese val dataset:", len(ch_valid_dataset))

    return ({"entrain":en_train_loader, "chtrain":ch_train_loader, 
            "envalid":en_valid_loader, "chvalid":ch_valid_loader}, en_text_proc, ch_text_proc, train_sampler)


def get_model(en_text_proc, ch_text_proc, args):
    en_sent_vocab = en_text_proc.vocab
    ch_sent_vocab = ch_text_proc.vocab

    enc = EncoderRNN(args.d_word_emb, len(en_sent_vocab), len(ch_sent_vocab),
        args.d_hidden)
    dec = DecoderRNN(args.d_word_emb, args.d_hidden)
    model = S2VTAttModel(enc, dec)

    # Initialize the networks and the criterion
    if len(args.start_from) > 0:
        print("Initializing weights from {}".format(args.start_from))
        model.load_state_dict(torch.load(args.start_from,
                                              map_location=lambda storage, location: storage))

    # Ship the model to GPU, maybe
    if torch.cuda.is_available():
        print("USING GPU")
        model.cuda()
        # if args.distributed:
          #   model.cuda()
            # model = torch.nn.parallel.DistributedDataParallel(model)
        # else:
          #   model = torch.nn.DataParallel(model).cuda()
        # elif torch.cuda.device_count() > 1:
          #   model = torch.nn.DataParallel(model).cuda()
        # else:
            # model.cuda()
    return model


def main(args):
    try:
        os.makedirs(args.checkpoint_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Checkpoint directory already exists.')
        else:
            raise

    print("GPU available?", torch.cuda.is_available())
    print('loading dataset')
    loader_dict, en_text_proc, ch_text_proc, train_sampler = get_dataset(args)

    print('building model')
    model = get_model(en_text_proc, ch_text_proc, args)

    if args.mode == 'inference':
        for language in ["en", "ch"]:
            inference(model, language, loader_dict[language + "valid"])
        return

    # filter params that don't require gradient (credit: PyTorch Forum issue 679)
    # smaller learning rate for the decoder
    if args.optim == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.learning_rate, betas=(args.alpha, args.beta), eps=args.epsilon)
        # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.learning_rate,
            weight_decay=1e-5,
            momentum=args.alpha,
            nesterov=True
        )
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum = 0.899999976158,
                              weight_decay=0.000500000023749)
    else:
        raise NotImplementedError

    # learning rate decay every 1 epoch
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.reduce_factor,
    #                                            patience=args.patience_epoch,
    #                                            verbose=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Number of parameter blocks in the network
    print("# of param blocks: {}".format(str(len(list(model.parameters())))))

    best_loss = float('inf')

    all_valid_losses = []
    all_training_losses = []

    curr_lang = "en"
    text_proc = None
    for train_epoch in range(args.max_epochs):

        t_epoch_start = time.time()

        # switch language every other epoch
        if train_epoch % 2 == 0:
            curr_lang = "en"
            text_proc = en_text_proc
        else:
            curr_lang = "ch"
            text_proc = ch_text_proc

        print('Epoch: {} out of {}, language: {}'.format(train_epoch + 1, 
            args.max_epochs, curr_lang))

        if args.distributed:
            train_sampler.set_epoch(train_epoch)

        epoch_loss = train(train_epoch, model, optimizer,
                           curr_lang, loader_dict[curr_lang + "train"],
                           len(text_proc.vocab), args)
        all_training_losses.append(epoch_loss)

        valid_loss = valid(model, curr_lang,
                           loader_dict[curr_lang + "valid"],
                           text_proc,
                           logging)
        all_valid_losses.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                # torch.save(model.module.state_dict(), os.path.join(args.checkpoint_path, 'best_model.t7'))
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, 'best_model.t7'))
            print('*'*5)
            print('Better validation loss {:.4f} found, save model'.format(valid_loss))

        # save eval and train losses
        if (args.distributed and dist.get_rank() == 0) or not args.distributed:
            torch.save({'train_loss':all_training_losses,
                        'valid_loss':all_valid_losses}, os.path.join(args.checkpoint_path,
                            'model_losses.t7'))

        # learning rate decay
        scheduler.step(valid_loss)

        # validation/save checkpoint every few epochs
        if train_epoch%args.save_checkpoint_every == 0 or train_epoch == args.max_epochs:
            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                # torch.save(model.module.state_dict(),
                torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_path, 'model_epoch_{}.t7'.format(train_epoch)))

        # all other process wait for the 1st process to finish
        # if args.distributed:
        #     dist.barrier()

        print('-'*80)
        print('Epoch {} summary'.format(train_epoch + 1))
        print('Train loss: {:.4f}, val loss: {:.4f}, Time: {:.4f}s'.format(
            epoch_loss, valid_loss, time.time()-t_epoch_start
        ))

        logging.info('Epoch {} summary'.format(train_epoch + 1))
        logging.info('Train loss: {:.4f}, val loss: {:.4f}, Time: {:.4f}s'.format(
            epoch_loss, valid_loss, time.time()-t_epoch_start
        ))

        print('-'*80)

### Training the network ###
def train(epoch, model, optimizer, language, train_loader, len_vocab, args):
    model.train() # training mode
    train_loss = []
    nbatches = len(train_loader)
    print("len trainloader:", nbatches)
    t_iter_start = time.time()

    loss = torch.nn.MSELoss()

    epoch_loss = 0
    for train_iter, data in enumerate(train_loader):        

        (img_batch, sentence_batch, captions, video_prefixes, lengths) = data
        # img_batch = Variable(img_batch)
        # sentence_batch = Variable(sentence_batch)

        t_model_start = time.time()

        if torch.cuda.is_available():
            img_batch, sentence_batch = img_batch.cuda(), sentence_batch.cuda()
        
        y_out = model(language, sentence_batch, lengths, img_batch, mode='train')

        flat_img_batch = torch.flatten(img_batch, start_dim=1)
        batch_loss = loss(y_out, flat_img_batch)
        # if scst_loss is not None:
        #     scst_loss *= args.scst_weight
        #     total_loss += scst_loss
        epoch_loss += batch_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        t_model_end = time.time()

        logging.info('\niter: [{}/{}], training loss: {:.4f}, '
              'data time: {:.4f}s, total time: {:.4f}s\n'.format(
            train_iter, nbatches, batch_loss.data.item(),
            t_model_start - t_iter_start,
            t_model_end - t_iter_start
        ))
        t_iter_start = time.time()

    epoch_loss = epoch_loss/nbatches

    return epoch_loss


### Validation ##
def valid(model, language, loader, text_proc, logger):
    model.eval()

    epoch_loss = 0

    gts = {}
    res = {}

    nbatches = len(loader)
    t_iter_start = time.time()
    loss = torch.nn.MSELoss()

    for val_iter, data in enumerate(loader):
        
        with torch.no_grad():
            (img_batch, sentence_batch, captions, video_prefixes, lengths) = data
            
            len_captions = len(captions[0])
            # img_batch = Variable(img_batch)
            # sentence_batch = Variable(sentence_batch)

            if torch.cuda.is_available():
                img_batch, sentence_batch = img_batch.cuda(), sentence_batch.cuda()

            t_model_start = time.time()
            y_out = model(language, sentence_batch, lengths, mode='inference')

            flat_img_batch = torch.flatten(img_batch, start_dim=1)

            batch_loss = loss(y_out, flat_img_batch)
            epoch_loss += batch_loss.item()

            t_model_end = time.time()

            logging.info('\niter: [{}/{}], validation loss: {:.4f}, '
                  'data time: {:.4f}s, total time: {:.4f}s\n'.format(
                val_iter, nbatches, batch_loss.data.item(),
                t_model_start - t_iter_start,
                t_model_end - t_iter_start
            ))
            t_iter_start = time.time()

        epoch_loss = epoch_loss/nbatches

    return epoch_loss

def inference(model, language, loader):
    print("Inference for {} language".format(language))
    model.eval()

    img_feats = []
    img_dim = 32
    vid_names = {}
    vid_ctr = 0
    for split in ["train","val"]:
        fpath = ""
        if split == "train":
            fpath = args.train_data_folder
        else:
            fpath = args.val_data_folder
        for i, vid in enumerate(os.listdir(fpath)):
            vid_names[vid_ctr] = os.path.join(split, vid)
            img_feat = np.squeeze(np.load(os.path.join(fpath, vid)))
            if img_feat.shape[0] != img_dim:
                odd_size = img_dim - img_feat.shape[0]
                img_feat = np.pad(img_feat,
                                  ((0, odd_size), (0, 0)),
                                  mode="constant",
                                  constant_values=0)
            img_feat = img_feat.reshape((img_dim*args.image_feat_size))
            img_feats.append(img_feat)
            vid_ctr += 1

    num_feats = 32*args.image_feat_size
    t = AnnoyIndex(num_feats, 'euclidean')
    if len(args.load_nn) > 0:
        print("loading nearest neighbors")
        t.load(args.load_nn, prefault=True)
    else:
        print("creating nearest neighbors")
        t.set_seed(42)
        for i, row in enumerate(img_feats):
            t.add_item(i, row)
        t.build(args.num_trees)
        t.save(args.save_nn)

    for val_iter, data in enumerate(loader):
        
        with torch.no_grad():
            (img_batch, sentence_batch, captions, video_prefixes, lengths) = data
            
            # img_batch = Variable(img_batch)
            # sentence_batch = Variable(sentence_batch)

            if torch.cuda.is_available():
                img_batch, sentence_batch = img_batch.cuda(), sentence_batch.cuda()

            y_out = model(language, sentence_batch, lengths, mode='inference')

            for i, row in enumerate(y_out):
                # print("query caption: {}".format(captions[i]))
                # print("ground truth video: {}".format(video_prefixes[i]))
                trunc_row = row[1024:]
                neighbors = t.get_nns_by_vector(trunc_row, args.num_neighbors)
                for neighbor in neighbors:
                    print(vid_names[neighbor], np.linalg.norm(t.get_item_vector(neighbor) - trunc_row.cpu().numpy()))

                printt()
                logging.info('query caption: {}\n'
                      'ground truth video: {}\n'
                      '1 nearest neighbor: {}\n'
                      '2 nearest neighbor: {}\n'
                      '3 nearest neighbor: {}\n'
                      '4 nearest neighbor: {}\n'
                      '5 nearest neighbor: {}\n'.format(
                    captions[i],
                    video_prefixes[i],
                    vid_names[neighbors[0]],
                    vid_names[neighbors[1]],
                    vid_names[neighbors[2]],
                    vid_names[neighbors[3]],
                    vid_names[neighbors[4]]
                ))


if __name__ == "__main__":
    main(args)
