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
from utils import *

def saveAttention(att, save_dir, offset, suffix=""):
  #np.savetxt(os.path.join(save_dir, "attention" + str(offset) + suffix + ".dat"), att, fmt="%f")
  plt.figure()
  plt.title("Attention at Offset " + str(offset))
  plt.imshow(att, interpolation="none", vmin=0., vmax=1., cmap="gray")
  plt.savefig(os.path.join(save_dir, "attention" + str(offset) + suffix + ".png"), dpi=600)
  plt.close()

def saveTranslation(batch_in, batch_out, prediction, save_dir, offset, suffix, dh):
  with open(os.path.join(save_dir, "translations_out_" + str(offset) + suffix + ".txt"), "w") as f:
    f.write("input:      " + dh.oneHotsToWords(batch_in[-1], dh.dict_token_to_word_langs[0]) + "\n")
    f.write("target:     " + dh.oneHotsToWords(batch_out[-1], dh.dict_token_to_word_langs[1]) + "\n")
    f.write("prediction: " + dh.softmaxesToWords(prediction[-1], dh.dict_token_to_word_langs[1], no_unk=False) + "\n")
    f.write("top 5:      " + "\n".join([str(d) for d in dh.topKPredictions(prediction[-1], 5, dh.dict_token_to_word_langs[1])]) + "\n")

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

  parser.add_argument("-m", "--maxlinelength",
    required    = False,
    default     = 51,
    type        = int,
    help        = "The maximum sequence length of a batch retrieved from the dataset.")

  parser.add_argument("--starttokenonoutput",
    required    = False,
    default     = False,
    action      = 'store_true',
    help        = "Boolean indicating that a \"<start>\" token should be at the beginning of translated strings.")

  parser.add_argument("--starttokenoninput",
    required    = False,
    default     = False,
    action      = 'store_true',
    help        = "Boolean indicating that a \"<start>\" token should be at the beginning of input strings.")

  parser.add_argument("-tf", "--teacherforcing",
    required    = False,
    default     = False,
    action      = 'store_true',
    help        = "Indicates whether or not teacher forcing should be used during training.")

  parser.add_argument("-kp", "--dropoutkeepprob",
    required    = False,
    default     = 0.8,
    type        = float,
    help        = "The dropout keep probability to use in recurrent layers of the network during training.")


  parser.add_argument("-tr", "--truncatelines",
    required    = False,
    default     = False,
    action      = 'store_true',
    help        = "When this flag is set, any training batch with a sequence length greater than the max line \
                   length will be truncated. If this flag is not set, new batches are retrieved until one is found \
                   that matches the max sequence length.")

  args = parser.parse_args()

  dh = DataHandler(args.englishtext, args.englishdict, args.targettext, args.targetdict, output_has_start_token=args.starttokenonoutput, input_has_start_token=args.starttokenoninput)
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

  repeated_batch = dh.getValidateBatch(1)
  while repeated_batch[1].shape[1] < 15:
    repeated_batch = dh.getValidateBatch(1)

  i = 0
  while (dh.num_epochs_elapsed < args.numepochs):
    curr_epoch_count = dh.num_epochs_elapsed
    if prev_epoch_count < curr_epoch_count:
      print("\n\n\n       new epoch!        ", curr_epoch_count)
      print("       saving the model        \n\n\n", curr_epoch_count)
      prev_epoch_count = dh.num_epochs_elapsed
      nmt.saveParams(os.path.join(save_dir, "nmt_checkpoint"), i)

    new_batch = dh.getTrainBatch(args.batchsize)
    if args.truncatelines:
      if new_batch[0].shape[1] > args.maxlinelength:
        new_batch[0] = new_batch[0][:, 0:args.maxlinelength, :]
        new_batch[1] = new_batch[1][:, 0:args.maxlinelength, :]
    else:
      while new_batch[0].shape[1] > args.maxlinelength:
        #print("That batch was too big, getting another one.")
        new_batch = dh.getTrainBatch(args.batchsize)

    loss, _ = nmt.trainStep(new_batch[0], new_batch[1], args.dropoutkeepprob, args.teacherforcing)

    if np.isnan(loss):
      print("Found a loss that is nan... exiting.")
      sys.exit(-1)
    loss_file.write(str(loss) + "," + str(new_batch[0].shape[1]) + "\n")

    if (i % 50) == 0:
      predictions, attention = nmt.predict(new_batch[0])
      saveTranslation(new_batch[0], new_batch[1], predictions, save_dir, i, "_train", dh)
      english_labels = dh.oneHotsToWords(new_batch[0][-1], dh.dict_token_to_word_langs[0])
      french_labels = dh.softmaxesToWords(predictions[-1], dh.dict_token_to_word_langs[1], no_unk=False)
      #saveAttention(attention[-1], save_dir, i, suffix="_train")
      saveAttentionMatrix(attention[-1], save_dir, i, english_labels.split(), french_labels.split(), suffix="_train")

      valid_batch = dh.getValidateBatch(1)
      predictions, attention = nmt.predict(valid_batch[0])
      #print("shape of predictions: ", predictions.shape)
      saveTranslation(valid_batch[0], valid_batch[1], predictions, save_dir, i, "", dh)
      english_labels_valid = dh.oneHotsToWords(valid_batch[0][-1], dh.dict_token_to_word_langs[0])
      french_labels_valid = dh.softmaxesToWords(predictions[-1], dh.dict_token_to_word_langs[1], no_unk=False)
      #saveAttention(attention[-1], save_dir, i)
      saveAttentionMatrix(attention[-1], save_dir, i, english_labels_valid.split(), french_labels_valid.split())

      #repeated_batch = dh.getValidateBatch(1)
      predictions, attention = nmt.predict(repeated_batch[0])
      #print("shape of predictions: ", predictions.shape)
      saveTranslation(repeated_batch[0], repeated_batch[1], predictions, save_dir, i, "_repeated", dh)
      english_labels_valid = dh.oneHotsToWords(repeated_batch[0][-1], dh.dict_token_to_word_langs[0])
      french_labels_valid = dh.softmaxesToWords(predictions[-1], dh.dict_token_to_word_langs[1], no_unk=False)
      #saveAttention(attention[-1], save_dir, i)
      saveAttentionMatrix(attention[-1], save_dir, i, english_labels_valid.split(), french_labels_valid.split(), suffix="_repeated")

      current_time = time.time()
      hours, rem = divmod(current_time - start_time, 3600)
      minutes, seconds = divmod(rem, 60)
      print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
      print("Num epochs: ", curr_epoch_count)
      print("Loss: ", loss)

#    if (i % 500) == 0:
#      nmt.saveParams(os.path.join(save_dir, "nmt_checkpoint"), i)
    i += 1
  nmt.saveParams(os.path.join(save_dir, "nmt_checkpoint"), i)
  loss_file.close()

if __name__ == "__main__":
  main(sys.argv)