import json
import os

import numpy as np

import misc_utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset
from rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel
from torch import nn
from torch.utils.data import DataLoader


def train(loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    model.train()
    #model = nn.DataParallel(model)
    for epoch in range(opt["epochs"]):
        lr_scheduler.step()

        iteration = 0
        #init_cider_scorer(opt["cached_tokens"])

        for data in loader:
            #torch.cuda.synchronize()
            i3d_feats = data['i3d_feats'] #.cuda()
            labels = data['labels'] #.cuda()
            masks = data['masks'] #.cuda()

            optimizer.zero_grad()

            seq_probs, _ = model(i3d_feats, labels, 'train')
            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            '''
            seq_probs, seq_preds = model(i3d_feats, mode='inference', opt=opt)
            reward = get_self_critical_reward(model, i3d_feats, data, seq_preds)
            print(reward.shape)
            loss = rl_crit(seq_probs, seq_preds, torch.from_numpy(reward).float()) #.cuda()
            '''
            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            #torch.cuda.synchronize()
            iteration += 1

            if not sc_flag:
                print("iter %d (epoch %d), train_loss = %.6f" %
                      (iteration, epoch, train_loss))
            else:
                print("iter %d (epoch %d), avg_reward = %.6f" %
                      (iteration, epoch, np.mean(reward[:, 0])))

        if epoch % opt["save_checkpoint_every"] == 0:
            model_path = os.path.join(opt["checkpoint_path"],
                                      'model_%d.pth' % (epoch))
            model_info_path = os.path.join(opt["checkpoint_path"],
                                           'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))


def main(opt):
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    opt["vocab_size"] = dataset.get_vocab_size()
    encoder = EncoderRNN(
        opt["dim_vid"],
        opt["dim_hidden"],
        bidirectional=bool(opt["bidirectional"]),
        input_dropout_p=opt["input_dropout_p"],
        rnn_cell=opt['rnn_type'],
        rnn_dropout_p=opt["rnn_dropout_p"])
    decoder = DecoderRNN(
        opt["vocab_size"],
        opt["max_len"],
        opt["dim_hidden"],
        opt["dim_word"],
        input_dropout_p=opt["input_dropout_p"],
        rnn_cell=opt['rnn_type'],
        rnn_dropout_p=opt["rnn_dropout_p"],
        bidirectional=bool(opt["bidirectional"]))
    model = S2VTAttModel(encoder, decoder)
    #model = S2VTModel(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"], opt['dim_vid'], rnn_cell=opt['rnn_type'], n_layers=opt['num_layers'], rnn_dropout_p=opt["rnn_dropout_p"])
    #model = model.cuda()
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    main(opt)