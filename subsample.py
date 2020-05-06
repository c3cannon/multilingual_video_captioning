import numpy as np
import os
import json
from random import sample, seed
import shutil
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data', default='', type=str, required=True)
parser.add_argument('--dest', default='', type=str, required=True)
parser.add_argument('--train_json', default='', type=str, required=True)
parser.add_argument('--val_json', default='', type=str, required=True)
parser.add_argument('--frac', default=0.25, type=float, required=False)

args = parser.parse_args()

if not os.path.exists(args.dest):
	os.mkdir(args.dest)

def make_sample(data_dir, json_file, frac, dest, stype):
	with open(json_file, 'r') as f:
		sentences = json.load(f)
	seed(42)
	subsample = sample(sentences, round(frac*len(sentences)))
	vid_ids = set([s["videoID"] + ".npy" for s in subsample])

	fpath = os.path.join(dest, stype)
	if not os.path.exists(fpath):
		os.mkdir(fpath)
	for vid in vid_ids:
	    shutil.copyfile(os.path.join(data_dir, "val", vid),
	    	os.path.join(fpath, vid))

	subsample_json = [x for x in sentences if x["videoID"] + ".npy" in vid_ids]
	with open(os.path.join(dest, "vatex_subsample_{}_v1.0.json".format(stype)), "w") as f:
	    json.dump(subsample_json, f)


def main(args):

	print("Writing training subsample...")
	make_sample(args.data, args.train_json, args.frac, args.dest, "train")

	print("Writing validation subsample...")
	make_sample(args.data, args.val_json, args.frac, args.dest, "val")

if __name__ == "__main__":
    main(args)
