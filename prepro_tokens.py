
import os
import json
import argparse
from six.moves import cPickle, xrange
from collections import defaultdict
import pdb

def precook(s, n=4, out=False):
  """
  Takes a string as input and returns an object that can be given to
  either cook_refs or cook_test. This is optional: cook_refs and cook_test
  can take string arguments as well.
  :param s: string : sentence to be converted into ngrams
  :param n: int    : number of ngrams for which representation is calculated
  :return: term frequency vector for occuring ngrams
  """
  words = s.split()
  counts = defaultdict(int)
  for k in xrange(1,n+1):
    for i in xrange(len(words)-k+1):
      ngram = tuple(words[i:i+k])
      counts[ngram] += 1
  return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def create_crefs(refs):
  crefs = []
  for ref in refs:
    # ref is a list of 5 captions
    crefs.append(cook_refs(ref))
  return crefs

def compute_doc_freq(crefs):
  '''
  Compute term frequency for reference data.
  This will be used to compute idf (inverse document frequency later)
  The term frequency is stored in the object
  :return: None
  '''
  document_frequency = defaultdict(float)
  for refs in crefs:
    # refs, k ref captions of one image
    for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
      document_frequency[ngram] += 1
      # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
  return document_frequency

def build_dict(videos, wtoi, params):
  wtoi['<eos>'] = 0

  count_vid = 0
  refs_words = []
  refs_idxs = []
  for vid in videos:
    ref_words = []
    ref_idxs = []
    for cap in videos[vid]['final_captions']:
      tmp_tokens = cap
      tmp_tokens = [_ if _ in wtoi else 'UNK' for _ in tmp_tokens]
      ref_words.append(' '.join(tmp_tokens))
      ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))

    refs_words.append(ref_words)
    refs_idxs.append(ref_idxs)
    count_vid += 1
  print('total videos:', count_vid)

  ngram_words = compute_doc_freq(create_crefs(refs_words))
  ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
  return ngram_words, ngram_idxs, count_vid

def main(params):

  videos = json.load(open(params['input_json'], 'r'))
  itow = json.load(open(params['info_json'], 'r'))['ix_to_word']
  wtoi = {w:i for i,w in itow.items()}

  ngram_words, ngram_idxs, ref_len = build_dict(videos, wtoi, params)
  #cPickle.dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(params['output_pkl']+'-words.p','w'), protocol=cPickle.HIGHEST_PROTOCOL)
  cPickle.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open(params['output_pkl']+'-idxs.p','wb'), protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  # input json
  parser.add_argument('--input_json', default='caption_train_english.json', help='input json file to process into hdf5')
  parser.add_argument('--info_json', default='info_train_english.json', help='info json file')
  parser.add_argument('--output_pkl', default='vatex_training', help='output pickle file')
  #parser.add_argument('--split', default='train', help='test, val, train, all')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  main(params)