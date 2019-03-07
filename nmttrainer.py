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

def saveAttention(att, save_dir, offset, suffix=""):
  #np.savetxt(os.path.join(save_dir, "attention" + str(offset) + suffix + ".dat"), att, fmt="%f")
  plt.figure()
  plt.title("Attention at Offset " + str(offset))
  plt.imshow(att, interpolation="none", vmin=0., vmax=1., cmap="gray")
  plt.savefig(os.path.join(save_dir, "attention" + str(offset) + suffix + ".png"), dpi=600)
  plt.close()

def saveTranslation(batch_in, batch_out, prediction, save_dir, offset, suffix, dh):
  with open(os.path.join(save_dir, "translations_out_" + str(offset) + suffix + ".txt"), "w") as f:
    f.write("input:      " + dh.oneHotsToWords(batch_in[0], dh.dict_token_to_word_langs[0]) + "\n")
    f.write("target:     " + dh.oneHotsToWords(batch_out[0], dh.dict_token_to_word_langs[1]) + "\n")
    f.write("prediction: " + dh.softmaxesToWords(prediction, dh.dict_token_to_word_langs[1], no_unk=False) + "\n")
    f.write("top 5:      " + str(dh.topKPredictions(prediction, 5, dh.dict_token_to_word_langs[1])) + "\n")

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
    default     = 80,
    help        = "The size of the training batches to use.")

  parser.add_argument("-n", "--numepochs",
    required    = True,
    type        = int,
    help        = "The number of times to pass through the training data.")

  parser.add_argument("-c", "--loadconfig",
    required    = False,
    default     = None,
    help        = "The location of a configuration file to load and continue training \
                   (e.g. some saved parameters)")

  args = parser.parse_args()

  dh = DataHandler(args.englishtext, args.englishdict, args.targettext, args.targetdict)
  sess = tf.Session()
  nmt = NMT(sess, in_vocab_size=dh.vocab_sizes[0], out_vocab_size=dh.vocab_sizes[1])
  sess.run(tf.global_variables_initializer())
  if args.loadconfig != None:
    nmt.loadParams(args.loadconfig)

  save_dir = os.path.expanduser("~/Documents/nmt_training_output/nmt_") + str(datetime.datetime.today()).replace(":", "-").replace(" ", "-")
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  loss_file = open(os.path.join(save_dir, "losses.dat"), "w")
  args_file = open(os.path.join(save_dir, "args.dat"), "w")
  json.dump(vars(args), args_file)
  args_file.close()

  start_time = time.time()
  prev_epoch_count = dh.num_epochs_elapsed
  #for i in range(120000):
  i = 0
  while (dh.num_epochs_elapsed < args.numepochs):
    curr_epoch_count = dh.num_epochs_elapsed
    if prev_epoch_count > curr_epoch_count:
      print("\n\n\n       new epoch!        \n\n\n", curr_epoch_count)
      prev_epoch_count = dh.num_epochs_elapsed
    new_batch = dh.getTrainBatch(args.batchsize)
    while new_batch[0].shape[1] > 50:
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
      predictions, attention = nmt.predict(new_batch[0])
      saveTranslation(new_batch[0], new_batch[1], predictions[0], save_dir, i, "_train", dh)
      saveAttention(attention[0], save_dir, i, suffix="_train")
      valid_batch = dh.getValidateBatch(1)
      predictions, attention = nmt.predict(valid_batch[0])
      print("shape of predictions: ", predictions.shape)
      saveTranslation(valid_batch[0], valid_batch[1], predictions[0], save_dir, i, "", dh)
      saveAttention(attention[0], save_dir, i)
      current_time = time.time()
      hours, rem = divmod(current_time - start_time, 3600)
      minutes, seconds = divmod(rem, 60)
      print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
      print("Num epochs: ", curr_epoch_count)
    if (i % 500) == 0:
      nmt.saveParams(os.path.join(save_dir, "nmt_checkpoint"), i)
    i += 1
  nmt.saveParams(os.path.join(save_dir, "nmt_checkpoint"), i)
  loss_file.close()

if __name__ == "__main__":
  main(sys.argv)