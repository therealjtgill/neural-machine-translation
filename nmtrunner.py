import argparse
from nmt import NMT
import numpy as np
from tokenconverter import *

if __name__ == "__main__":
  parser = argparse.ArgumentParser(descrption="Opens up the saved weights for \
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
  
  args = parser.parse_args()
  sess = tf.Session()
  eng_vocab_size = len(args.englishdict)
  tar_vocab_size = len(args.targetdict)
  eng_words_to_tokens = args.englishdict
  eng_tokens_to_words = {v:k for k, v in args.englishdict.items()}
  tar_words_to_tokens = args.targetdict
  tar_tokens_to_words = {v:k for k, v in args.targetdict.items()}
  nmt = NMT(sess, in_vocab_size=eng_vocab_size, out_vocab_size=tar_vocab_size)

  input_str = args.stringtotranslate.strip().replace(".", "")
  input_tokens = [eng_words_to_tokens[w] if w in eng_words_to_tokens else eng_words_to_tokens["<unk>"]
                  for w in input_str.split(" ")]
  print(input_tokens)