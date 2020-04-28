import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from dataloader import VideoDataset
import misc.utils as utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer
from pandas.io.json import json_normalize
import val_opts


def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["enCap"], data_frame["videoID"]):
        for cap in row[0]:
            if row[1] in gts:
                gts[row[1]].append(
                    {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': cap})
            else:
                gts[row[1]] = []
                gts[row[1]].append(
                    {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': cap})
    
    with open(os.path.join("gts_test.csv"), 'w') as gts_table:
        gts_table.write(json.dumps(gts))

    return gts


def test(model, crit, dataset, vocab, opt):
    model.eval()
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    scorer = COCOScorer()
    '''
    gt_dataframe = json_normalize(
        json.load(open('test_videodatainfo.json'))['sentences'])
    '''
    gt_dataframe = json_normalize(
        json.load(open('vatex_validation.json')))
    #print(gt_dataframe)
    gts = convert_data_to_coco_scorer_format(gt_dataframe)

    results = []
    samples = {}
    for data in loader:
        # forward the model to get loss
        i3d_feats = data['i3d_feats'].squeeze(1) #.cuda()
        labels = data['labels'] #.cuda()
        masks = data['masks'] #.cuda()
        video_ids = data['video_ids']

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_probs, seq_preds = model(
                i3d_feats, mode='inference', opt=opt)

        sents = utils.decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]

    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
    results.append(valid_score)
    print(valid_score)

    if not os.path.exists(opt["results_path"]):
        os.makedirs(opt["results_path"])

    with open(os.path.join(opt["results_path"], "epoch10_scores.txt"), 'a') as scores_table:
        scores_table.write(json.dumps(results[0]) + "\n")
    with open(os.path.join(opt["results_path"],
                           opt["model"].split("/")[-1].split('.')[0] + "_epoch10.json"), 'w') as prediction_results:
        json.dump({"predictions": samples, "scores": valid_score},
                  prediction_results, indent=2)

def main(opt):
    dataset = VideoDataset(opt, 'test', 'english')
    opt["vocab_size"] = 22114 #28000 dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len
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
    # Setup the model
    model.load_state_dict(torch.load(opt["saved_model"]))
    crit = utils.LanguageModelCriterion()

    test(model, crit, dataset, dataset.get_vocab(), opt)


if __name__ == '__main__':
    opt = val_opts.parse_opt()
    opt = vars(opt)

    main(opt)