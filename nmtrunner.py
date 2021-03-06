import argparse
import nmt
from nmt import NMT
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import os
import tensorflow as tf
from tokenconverter import *
from utils import *

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Opens up the saved weights for \
    a given encoder/decoder/attention, applies them to a model, and \
    translates English to French.")

  parser.add_argument("-ed", "--englishdict",
    required    = True,
    help        = "The location of the dictionary that translates english words to tokens.")

  parser.add_argument("-td", "--targetdict",
    required    = True,
    help        = "The location of the dictionary that translates target language words to tokens.")

  parser.add_argument("-c", "--loadconfig",
    required    = True,
    default     = None,
    help        = "The location of a configuration file to load and continue training \
                   (e.g. some saved parameters).")

  parser.add_argument("-s", "--stringtotranslate",
    required    = True,
    default     = None,
    help        = "The string that will be translated into the target language.")

  parser.add_argument("--usecpu",
    required    = False,
    default     = False,
    action      = 'store_true',
    help        = "Boolean flag indicating that the computation graph should be executed on the CPU.")
  
  args = parser.parse_args()
  if args.usecpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  sess = sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  # load JSON from dictionary...
  d1 = open(args.englishdict)
  eng_words_to_tokens = json.load(d1)
  d1.close()
  eng_tokens_to_words = {v:k for k, v in eng_words_to_tokens.items()}
  d2 = open(args.targetdict)
  tar_words_to_tokens = json.load(d2)
  d2.close()
  tar_tokens_to_words = {v:k for k, v in tar_words_to_tokens.items()}

  eng_vocab_size = len(eng_words_to_tokens)
  tar_vocab_size = len(tar_words_to_tokens)

  nmt = NMT(sess, in_vocab_size=eng_vocab_size, out_vocab_size=tar_vocab_size, train=False)
  nmt.loadParams(args.loadconfig)

  input_str = args.stringtotranslate.strip()
  input_tokens = [eng_words_to_tokens[w] if w in eng_words_to_tokens else eng_words_to_tokens["<unk>"]
                  for w in input_str.split(" ")]
  print(input_tokens)
  input_one_hots = tokensToOneHots(input_tokens, eng_vocab_size)
  print("input one hots shape: ", input_one_hots.shape)
  input_hots = [a.argmax() for a in input_one_hots]
  print("hot indices: ", input_hots)
  
  # Actually run the input tokens through the network and get the translation...

  print("string to translate: ", args.stringtotranslate)
  predictions = nmt.predict([input_one_hots])
  print("shape of predictions", predictions[0].shape)
  predicted_words = softmaxesToWords(predictions[0][0], tar_tokens_to_words, no_unk=False)
  print(" ".join(predicted_words))
  greedy_hot_indices, greedy_attention = nmt.greedySearch([input_one_hots])
  print("greedy search output: ", greedy_hot_indices)
  greedy_tokens = [i + 1 for i in greedy_hot_indices]
  greedy_words = tokensToWords(greedy_tokens, tar_tokens_to_words)
  greedy_string = " ".join(greedy_words)
  print("len greedy words: ", len(greedy_words))
  print("greedy translation: ", greedy_string)
  
#  for topk in topKPredictions(predictions[0][0], 5, tar_tokens_to_words):
#    print(topk)
  plotAttentionMatrix(predictions[1][0], ".", input_str.split(), predicted_words)
  plotAttentionMatrix(greedy_attention, ".", input_str.split(), greedy_words)
