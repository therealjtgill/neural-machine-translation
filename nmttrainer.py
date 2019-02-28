import argparse
import datetime
import json
import matplotlib.pyplot as plt
from nmt import NMT
import numpy as np
import os
import sys
import tensorflow as tf
from tokenizedloader import DataHandler

def saveAttention(att, save_dir, offset):
  np.savetxt(os.path.join(save_dir, "attention" + str(offset) + ".dat"), att, fmt="%f")
  plt.figure()
  plt.title("Attention at Offset " + str(offset))
  plt.imshow(att, interpolation="none", vmin=0., vmax=1., cmap="gray")
  plt.savefig(os.path.join(save_dir, "attention" + str(offset) + ".png"), dpi=600)
  plt.close()

def main(argv):
  parser = argparse.ArgumentParser(description="Script to train a Neural Machine Translation Model.\
    This is based off of the work by Bahdanau; an encoder/decoder model with attention.")

  parser.add_argument("-et", "--englishtext",
    required    = True,
    help        = "The location of tokenized english training text.")

  parser.add_argument("-tt", "--targettext",
    required    = True,
    help        = "The location of tokenized training text in the target language.")

  parser.add_argument("-ed", "--englishdict",
    required    = True,
    help        = "The location of the dictionary that translates english words to tokens.")

  parser.add_argument("-td", "--targetdict",
    required    = True,
    help        = "The location of the dictionary that translates target language words to tokens.")

  parser.add_argument("-b", "--batchsize",
    required    = False,
    default     = 20,
    help        = "The size of the training batches to use.")

  args = parser.parse_args()

  dh = DataHandler(args.englishtext, args.englishdict, args.targettext, args.targetdict)
  sess = tf.Session()
  nmt = NMT(sess, in_vocab_size=dh.vocab_sizes[0], out_vocab_size=dh.vocab_sizes[1])
  sess.run(tf.global_variables_initializer())

  losses = []
  save_dir = os.path.expanduser("~/Documents/nmt_training_output/nmt_") + str(datetime.datetime.today()).replace(":", "-").replace(" ", "-")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  for i in range(70000):
    new_batch = dh.getTrainBatch(args.batchsize)
    while new_batch[0].shape[1] > 120:
      print("That batch was too big, getting another one.")
      new_batch = dh.getTrainBatch(args.batchsize)
    loss, attention = nmt.trainStep(new_batch[0], new_batch[1])
    losses.append(loss)
    print("train on, mothafucka")
    if (i % 50) == 0:
      saveAttention(attention[0], save_dir, i)
  print(losses)
  np.savetxt(os.path.join(save_dir, "losses.dat"), losses, fmt="%f")

if __name__ == "__main__":
  main(sys.argv)