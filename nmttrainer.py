import argparse
import datetime
import json
import matplotlib.pyplot as plt
from nmt import NMT
import numpy as np
import os
import sys
import tensorflow as tf
import time
from tokenizedloader import DataHandler

def saveAttention(att, save_dir, offset):
  np.savetxt(os.path.join(save_dir, "attention" + str(offset) + ".dat"), att, fmt="%f")
  plt.figure()
  plt.title("Attention at Offset " + str(offset))
  plt.imshow(att, interpolation="none", vmin=0., vmax=1., cmap="gray")
  plt.savefig(os.path.join(save_dir, "attention" + str(offset) + ".png"), dpi=600)
  plt.close()

def saveTranslation(batch_in, batch_out, prediction, save_dir, offset, dh):
  with open(os.path.join(save_dir, "translations_out_" + str(offset) + ".txt"), "w") as f:
    f.write("input:      " + dh.oneHotsToWords(batch_in[0], dh.dict_token_to_word_langs[0]) + "\n")
    f.write("target:     " + dh.oneHotsToWords(batch_out[0], dh.dict_token_to_word_langs[1]) + "\n")
    f.write("prediction: " + dh.softmaxesToWords(prediction, dh.dict_token_to_word_langs[1]) + "\n")

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
    #default     = 30,
    default     = 45,
    help        = "The size of the training batches to use.")

  args = parser.parse_args()

  dh = DataHandler(args.englishtext, args.englishdict, args.targettext, args.targetdict)
  sess = tf.Session()
  nmt = NMT(sess, in_vocab_size=dh.vocab_sizes[0], out_vocab_size=dh.vocab_sizes[1])
  sess.run(tf.global_variables_initializer())

  save_dir = os.path.expanduser("~/Documents/nmt_training_output/nmt_") + str(datetime.datetime.today()).replace(":", "-").replace(" ", "-")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  loss_file = open(os.path.join(save_dir, "losses.dat"), "w")

  start_time = time.time()
  prev_epoch_count = dh.num_epochs_elapsed
  #for i in range(120000):
  i = 0
  while (dh.num_epochs_elapsed < 5):
    curr_epoch_count = dh.num_epochs_elapsed
    if prev_epoch_count > curr_epoch_count:
      print("\n\n\n       new epoch!        \n\n\n", curr_epoch_count)
      prev_epoch_count = dh.num_epochs_elapsed
    new_batch = dh.getTrainBatch(args.batchsize)
    while new_batch[0].shape[1] > 90:
      print("That batch was too big, getting another one.")
      new_batch = dh.getTrainBatch(args.batchsize)
    loss, _ = nmt.trainStep(new_batch[0], new_batch[1])
    print("loss: ", loss)
    if np.isnan(loss):
      print("Found a loss that is nan... exiting.")
      sys.exit(-1)
    print("train on, mothafucka")
    loss_file.write(str(loss) + "\n")
    if (i % 50) == 0:
      valid_batch = dh.getValidateBatch(1)
      predictions, attention = nmt.predict(valid_batch[0])
      print("shape of predictions: ", predictions.shape)
      saveTranslation(valid_batch[0], valid_batch[1], predictions[0], save_dir, i, dh)
      saveAttention(attention[0], save_dir, i)
      current_time = time.time()
      hours, rem = divmod(current_time - start_time, 3600)
      minutes, seconds = divmod(rem, 60)
      print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
      print("Num epochs: ", curr_epoch_count)
    if (i % 500) == 0:
      nmt.saveParams(os.path.join(save_dir, "nmt_checkpoint"), i)
    i += 1

if __name__ == "__main__":
  main(sys.argv)