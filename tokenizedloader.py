from __future__ import print_function

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

class DataHandler(object):

  def __init__(self, file_language_1, dict_language_1, file_language_2, dict_language_2):
    '''
    The dictionaries map words to tokens in a 1:1 relationship.
    The language files are sentence-aligned between the two languages. There is
    one sentence per line, file1.line(i) == file2.line(i) in translation space.
    '''
    self.file_lang_1 = file_language_1
    self.file_lang_2 = file_language_2
    self.dict_lang_1 = None
    self.dict_lang_2 = None

    with open(dict_language_1, "r") as d1:
      self.dict_lang_1 = json.load(d1)

    with open(dict_language_2, "r") as d2:
      self.dict_lang_2 = json.load(d2)

    self.vocab_size_1 = len(self.dict_lang_1)
    self.vocab_size_2 = len(self.dict_lang_2)
    print("language 1 vocab size: ", self.vocab_size_1)
    print("language 2 vocab size: ", self.vocab_size_2)
    self.num_lines_1 = 0
    self.num_lines_2 = 0

    self.lengths_1 = {}
    self.lengths_2 = {}

    self.train_data_1     = self.file_lang_1 + "_train"
    self.test_data_1      = self.file_lang_1 + "_test"
    self.validate_data_1  = self.file_lang_1 + "_validate"

    self.train_data_2     = self.file_lang_2 + "_train"
    self.test_data_2      = self.file_lang_2 + "_test"
    self.validate_data_2  = self.file_lang_2 + "_validate"

    self.preprocess(show_plot=True)

  def getSentenceLengths(self, filename):
    line_num = 0
    lengths = {}
    with open(filename, "r") as f:
      for line_num, line in enumerate(f):
        split_line = line.split(" ")
        line_length = len(split_line)
        if line_length in lengths:
          lengths[line_length].append(line_num)
        else:
          lengths[line_length] = [line_num,]

    return lengths, line_num

  def getTrainBatch(self, batch_size):
    return self.getBatch(batch_size, source="train")

  def getBatch(self, batch_size, source="train"):
    if source == "train":
      pass

  def preprocess(self, splits=[0.7, 0.15, 0.15], show_plot=False):

    self.lengths_1, self.num_lines_1 = self.getSentenceLengths(self.file_lang_1)

    self.lengths_2, self.num_lines_2 = self.getSentenceLengths(self.file_lang_2)

    print(self.num_lines_1, self.num_lines_2)      
    if self.num_lines_1 != self.num_lines_2:
      print("The two language files must be sentence-aligned, but they do not\
             have the same number of lines.",
             self.num_lines_1,
             self.num_lines_2)
      sys.exit(-1)

    if show_plot:
      # Plot the histogram of sentence lengths for both language files.
      plt.figure()
      x1 = [t for t in self.lengths_1]
      y1 = [len(self.lengths_1[t]) for t in x1]
      plt.bar(x1, y1)
      plt.figure()
      x2 = [t for t in self.lengths_2]
      y2 = [len(self.lengths_2[t]) for t in x2]
      plt.bar(x2, y2)
      plt.draw()

    file_indices = np.arange(self.num_lines_1)
    num_indices = len(file_indices)
    np.random.shuffle(file_indices)

    train_start     = 0
    num_train       = int(splits[0]*num_indices)
    train_indices   = file_indices[train_start:train_start + num_train]

    test_start      = train_start + num_train
    num_test        = int(splits[1]*num_indices)
    test_indices    = file_indices[test_start:test_start + num_test]

    validate_start  = test_start + num_test
    num_validate    = int(splits[2]*num_indices)
    validate_indices    = file_indices[validate_start:]

    print("num train sentences: ", len(train_indices))
    print("num test sentences: ", len(test_indices))
    print("num validate sentences: ", len(validate_indices))

    # Split language 1 file into training, test, validation sets.
    with open(self.file_lang_1, "r") as f1:
      if not os.path.exists(self.train_data_1):
        with open(self.train_data_1, "w") as trl1:
          for i, line in enumerate(f1):
            if i in train_indices:
              trl1.write(line)

      if not os.path.exists(self.test_data_1):
        f1.seek(0)
        with open(self.test_data_1, "w") as tel1:
          for j, line in enumerate(f1):
            if j in test_indices:
              tel1.write(line)

      if not os.path.exists(self.validate_data_1):
        f1.seek(0)
        with open(self.validate_data_1, "w") as val1:
          for k, line in enumerate(f1):
            if k in validate_indices:
              val1.write(line)

    # Split language 2 file into training, test, validation sets.
    with open(self.file_lang_2, "r") as f2:
      if not os.path.exists(self.train_data_2):
        with open(self.train_data_2, "w") as trl2:
          for i, line in enumerate(f2):
            if i in train_indices:
              trl2.write(line)

      if not os.path.exists(self.test_data_2):
        f2.seek(0)
        with open(self.test_data_2, "w") as tel2:
          for j, line in enumerate(f2):
            if j in test_indices:
              tel2.write(line)

      if not os.path.exists(self.validate_data_2):
        f2.seek(0)
        with open(self.validate_data_2, "w") as val2:
          for k, line in enumerate(f2):
            if k in validate_indices:
              val2.write(line)

def main(argv):
  parser = argparse.ArgumentParser(description="Loads tokenized language data \
    from sentence-aligned corpora in different languages.")

  parser.add_argument("-l1", "--language1",
    required    = True,
    help        = "The location of the file containing the tokenized sentences \
                   for the first language.")

  parser.add_argument("-d1", "--dict1",
    required    = True,
    help        = "The location of the dictionary that translates words to \
                   tokens in the first language.")

  parser.add_argument("-l2", "--language2",
    required    = True,
    help        = "The location of the file containing the tokenized sentences \
                   for the second language.")

  parser.add_argument("-d2", "--dict2",
    required    = True,
    help        = "The location of the dictionary that translates words to \
                   tokens in the second language.")

  args = parser.parse_args()

  if not os.path.exists(args.language1):
    print("Couldn't find tokenized file: ", args.language1)
    sys.exit(-1)

  if not os.path.exists(args.language2):
    print("Couldn't find tokenized file: ", args.language2)
    sys.exit(-1)

  if not os.path.exists(args.dict1):
    print("Couldn't find dictionary file: ", args.dict1)
    sys.exit(-1)

  if not os.path.exists(args.dict2):
    print("Couldn't find dictionary file: ", args.dict2)
    sys.exit(-1)

  dh = DataHandler(args.language1, args.dict1, args.language2, args.dict2)
  plt.show()

if __name__ == "__main__":
  main(sys.argv)