from __future__ import print_function

import argparse
import datetime
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
    self.file_langs = (file_language_1, file_language_2)
    self.dict_langs = [None, None]

    with open(dict_language_1, "r") as d1:
      self.dict_langs[0] = json.load(d1)

    with open(dict_language_2, "r") as d2:
      self.dict_langs[1] = json.load(d2)

    self.vocab_sizes = (len(self.dict_langs[0]), len(self.dict_langs[1]))

    print("language 1 vocab size: ", self.vocab_sizes[0])
    print("language 2 vocab size: ", self.vocab_sizes[1])
    self.num_file_lines = [0, 0]

    self.file_lang_lengths = [{}, {}]

    train_file_1     = self.file_langs[0] + "_train"
    test_file_1      = self.file_langs[0] + "_test"
    validate_file_1  = self.file_langs[0] + "_validate"

    train_file_2     = self.file_langs[1] + "_train"
    test_file_2      = self.file_langs[1] + "_test"
    validate_file_2  = self.file_langs[1] + "_validate"

    self.train_files = (train_file_1, train_file_2)
    self.test_files  = (test_file_1, test_file_2)
    self.validate_files = (validate_file_1, validate_file_2)

    # The train/test/validation files are all the same size (e.g. they all have
    # the same number of lines).
    self.train_files_size    = 0
    self.test_files_size     = 0
    self.validate_files_size = 0

    self.used_train_indices    = []
    self.used_test_indices     = []
    self.used_validate_indices = []

    # The number of batches that will be loaded when a request for a single
    # batch of data is received.
    self._preloaded_batch_depth = 20

    # Pull out several batches' worth of data at a time to avoid tons of file
    # file opens and reads. These wll be lists of raw strings that will be
    # tokenized while the batch is generated.
    self.preloaded_train_data    = [[], []]
    self.preloaded_test_data     = [[], []]
    self.preloaded_validate_data = [[], []]

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

    # Line numbers start at 0, which leaves the line count off by 1.
    return lengths, line_num + 1


  def preprocess(self, splits=[0.7, 0.15, 0.15], show_plot=False):

    for i in range(2):
      self.file_lang_lengths[i], self.num_file_lines[i] = self.getSentenceLengths(self.file_langs[i])

    print(self.num_file_lines[0], self.num_file_lines[1])
    if self.num_file_lines[0] != self.num_file_lines[1]:
      print("The two language files must be sentence-aligned, but they do not\
             have the same number of lines.",
             self.num_file_lines[0],
             self.num_file_lines[1])
      sys.exit(-1)

    if show_plot:
      # Plot the histogram of sentence lengths for both language files.      
      for i in range(2):
        plt.figure()
        x1 = [t for t in self.file_lang_lengths[i]]
        y1 = [len(self.file_lang_lengths[i][t]) for t in x1]
        plt.bar(x1, y1)
      plt.draw()

    file_indices = np.arange(self.num_file_lines[1])
    num_indices = len(file_indices)
    np.random.shuffle(file_indices)

    train_start      = 0
    num_train        = int(splits[0]*num_indices)
    train_indices    = file_indices[train_start:train_start + num_train]
    self.train_files_size = num_train

    test_start       = train_start + num_train
    num_test         = int(splits[1]*num_indices)
    test_indices     = file_indices[test_start:test_start + num_test]
    self.test_files_size = num_test

    validate_start   = test_start + num_test
    num_validate     = num_indices - num_train - num_test
    validate_indices = file_indices[validate_start:]
    self.validate_files_size = num_validate

    print("estimated num train sentences: ", num_train)
    print("estimated num test sentences: ", num_test)
    print("estimated num validate sentences: ", num_validate)

    # Get the list of lines numbers that should be ignored (e.g. lines in
    # either file that are blank)
    ignore_lines = []
    for n in range(2):
      with open(self.file_langs[n], "r") as f1:
        for i, line in enumerate(f1):
          if line.strip() == "":
            ignore_lines.append(i)
    ignore_lines = list(set(ignore_lines))

    # Split language files into training, test, validation sets.
    for n in range(2):
      with open(self.file_langs[n], "r") as f1:
        if not os.path.exists(self.train_files[n]):
          with open(self.train_files[n], "w") as trl1:
            f1.seek(0)
            for i, line in enumerate(f1):
              if i in train_indices and i not in ignore_lines:
                trl1.write(line)
        else:
          #if n == 0:
          _, self.train_files_size = self.getSentenceLengths(self.train_files[n])
          print(self.train_files_size)

        if not os.path.exists(self.test_files[n]):
          f1.seek(0)
          with open(self.test_files[n], "w") as tel1:
            for j, line in enumerate(f1):
              if j in test_indices and j not in ignore_lines:
                tel1.write(line)
        else:
          if n == 0:
            _, self.test_files_size = self.getSentenceLengths(self.test_files[n])

        if not os.path.exists(self.validate_files[n]):
          f1.seek(0)
          with open(self.validate_files[n], "w") as val1:
            f1.seek(0)
            for k, line in enumerate(f1):
              if k in validate_indices and k not in ignore_lines:
                val1.write(line)
        else:
          if n == 0:
            _, self.validate_files_size = self.getSentenceLengths(self.validate_files[n])

    print("num train sentences: ", self.train_files_size)
    print("num test sentences: ", self.test_files_size)
    print("num validate sentences: ", self.validate_files_size)


  def getTrainBatch(self, batch_size):
    return self.getBatch(batch_size, source="train")


  def getTestBatch(self, batch_size):
    return self.getBatch(batch_size, source="test")


  def getValidateBatch(self, batch_size):
    return self.getBatch(batch_size, source="validate")


  def rawTextToOneHots(self, lines, vocab):
    '''
    Expects lines to be an array of strings, with token characters delimited by
    spaces. So lines[i].split(" ") provides token numbers of the words in the
    line.
    Expects vocab to be a dictionary of words to token numbers.
    '''
    max_line_length = max([len(l.split(" ")) for l in lines]) + 1
    batch_size = len(lines)
    vocab_size = len(vocab)
    one_hots = np.zeros((batch_size, max_line_length, vocab_size))
    one_hots[:, :, vocab_size - 1] = 1.0

    for bs in range(batch_size):
      line_tokens = lines[bs].strip().split(" ")
      #print("line tokens: ", lines[bs], line_tokens)
      for sl in range(len(line_tokens)):
        hot_index = int(line_tokens[sl])
        one_hots[bs, sl, vocab_size - 1] = 0.0
        one_hots[bs, sl, hot_index] = 1.0

    return one_hots


  def getBatch(self, batch_size, source):
    batch_lines = [[], []]
    batch = [None, None]
    #batch_in = None
    #batch_out = None
    if source == "train":

      if len(self.preloaded_train_data[0]) != len(self.preloaded_train_data[1]):
        print("The two preloaded data items do not have the same number of items")
        sys.exit(-1)

      all_indices = np.arange(self.train_files_size)
      if (len(self.preloaded_train_data[0]) >= batch_size) and \
         (len(self.preloaded_train_data[1]) >= batch_size):
        batch_lines[0]  = self.preloaded_train_data[0][0:batch_size]
        batch_lines[1]  = self.preloaded_train_data[1][0:batch_size]
        del self.preloaded_train_data[0][0:batch_size]
        del self.preloaded_train_data[1][0:batch_size]
      else:
        num_sequences = batch_size*self._preloaded_batch_depth
        remaining_indices = list(set(all_indices).difference(set(self.used_train_indices)))
        print("Number of remaining indices: ", len(remaining_indices))
        # If the number of remaining indices drops below the number of lines
        # that must be loaded, then the used training indices is reset and we
        # start sampling from the entire training file.
        if len(remaining_indices) < num_sequences:
          print("Ran through the entire training set! Resetting the used training indices.")
          print("    This batch will have indices sampled from the entire dataset.")
          self.used_train_indices.clear()

        np.random.shuffle(remaining_indices)
        preloaded_indices = remaining_indices[0:num_sequences]
        self.used_train_indices += preloaded_indices
        for i in range(2):
          with open(self.train_files[i]) as traf:
            for j, line in enumerate(traf):
              #print("line in training file: ", j)
              #if len(batch_lines[i]) >= num_sequences:
              if len(self.preloaded_train_data[i]) == num_sequences:
                print("preloaded training data ", i, " has ", len(self.preloaded_train_data[i]), "things in it")
                break
              if j in preloaded_indices:
                #print("index-to-be added found!")
                print("indices remaining: ", num_sequences - len(self.preloaded_train_data[i]), "          \r", end="")
                self.preloaded_train_data[i].append(line)
          print("Finished loading batch indices from file ", self.train_files[i])

        batch_lines[0]  = self.preloaded_train_data[0][0:batch_size]
        batch_lines[1]  = self.preloaded_train_data[1][0:batch_size]
        del self.preloaded_train_data[0][0:batch_size]
        del self.preloaded_train_data[1][0:batch_size]

      for i, bstring in enumerate(batch_lines):
        batch[i] = self.rawTextToOneHots(bstring, self.dict_langs[i])
      return batch

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
  draw_times = []
  start_time = 0
  end_time = 0
  for i in range(21700):
    start_time = datetime.datetime.now()
    batch = dh.getTrainBatch(64)
    end_time = datetime.datetime.now()
    diff = end_time - start_time
    draw_times.append(diff.seconds)
    print("That batch took ", diff.seconds, " seconds to be drawn.")
    print(batch[0].shape, batch[1].shape)
  plt.show()

if __name__ == "__main__":
  main(sys.argv)