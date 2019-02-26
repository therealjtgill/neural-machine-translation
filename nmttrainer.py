import argarse
import json
import matplotlib.pyplot as plt
from nmt import NMT
import numpy as np
import os

def main(argv):
  parser = argparse.ArgumentParser(description="Script to train a Neural Machine Translation Model.\
    This is based off of the work by Bahdanau; an encoder/decoder model with attention.")

  parser.add_argument("-e", "--englishtext",
    required    = True,
    help        = "The location of tokenized english training text.")

  parser.add_argument("-t", "--targettext",
    required    = True,
    help        = "The location of tokenized training text in the target language.")

  parser.add_argument("-b", "--batchsize",
    required    = False,
    default     = 64,
    help        = "The size of the training batches to use.")

  args = parser.parse_args()

  

if __name__ == "__main__":
  main(sys.argv)