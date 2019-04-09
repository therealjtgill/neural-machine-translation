from __future__ import print_function

import argparse
import copy
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

class DataHandler(object):

  def __init__(self, file_language_1, dict_language_1, file_language_2, dict_language_2, output_has_start_token=True, input_has_start_token=True):
    '''
    The dictionaries map words to tokens in a 1:1 relationship.
    The language files are sentence-aligned between the two languages. There is
    one sentence per line, file1.line(i) == file2.line(i) in translation space.
    '''
    self.file_langs = (file_language_1, file_language_2)
    self.dict_word_to_token_langs = [None, None]
    self._output_has_start_token = output_has_start_token
    self._input_has_start_token = input_has_start_token

    with open(dict_language_1, "r") as d1:
      self.dict_word_to_token_langs[0] = json.load(d1)

    with open(dict_language_2, "r") as d2:
      self.dict_word_to_token_langs[1] = json.load(d2)

    self.dict_token_to_word_langs = [{}, {}]
    self.dict_token_to_word_langs[0] = {v: k for k, v in self.dict_word_to_token_langs[0].items()}
    self.dict_token_to_word_langs[1] = {v: k for k, v in self.dict_word_to_token_langs[1].items()}

    self.num_epochs_elapsed = 0

    self.vocab_sizes = (len(self.dict_word_to_token_langs[0]), len(self.dict_word_to_token_langs[1]))

    print("language 1 vocab size: ", self.vocab_sizes[0])
    print("language 2 vocab size: ", self.vocab_sizes[1])
    self.num_file_lines = [0, 0]

    self.file_lang_lengths = [{}, {}]

    train_file_1     = self.file_langs[0] + "_train_50"
    test_file_1      = self.file_langs[0] + "_test_50"
    validate_file_1  = self.file_langs[0] + "_validate_50"

    train_file_2     = self.file_langs[1] + "_train_50"
    test_file_2      = self.file_langs[1] + "_test_50"
    validate_file_2  = self.file_langs[1] + "_validate_50"

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
    '''
    Grab the counts of the number of sentences with given lengths and the total
    number of sentences in the file 'filename'.
    '''
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
    '''
    Create test, training, and validation sets. Eliminate lines from both files
    if either file has a blank line.
    '''
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
    train_indices    = list(file_indices[train_start:train_start + num_train])
    self.train_files_size = num_train

    test_start       = train_start + num_train
    num_test         = int(splits[1]*num_indices)
    test_indices     = list(file_indices[test_start:test_start + num_test])
    self.test_files_size = num_test

    validate_start   = test_start + num_test
    num_validate     = num_indices - num_train - num_test
    validate_indices = list(file_indices[validate_start:])
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
          if (n == 0) and len(line) > 51:
            ignore_lines.append(i)
    ignore_lines = list(set(ignore_lines))

    print("Ignoring ", len(ignore_lines), " lines of text.")

    print("len train indices: ", len(train_indices))
    train_indices_ = sorted(list(set(train_indices) - set(ignore_lines)))
    test_indices_ = sorted(list(set(test_indices) - set(ignore_lines)))
    validate_indices_ = sorted(list(set(validate_indices) - set(ignore_lines)))
    print("len train indices: ", len(train_indices))

    print("Finished restructuring the test/train/validate indices.")

    # Split language files into training, test, validation sets.
    for n in range(2):
      with open(self.file_langs[n], "r") as f1:
        train_indices = copy.deepcopy(train_indices_)
        if not os.path.exists(self.train_files[n]):
          with open(self.train_files[n], "w") as trl1:
            f1.seek(0)
            for i, line in enumerate(f1):
              if len(train_indices) == 0:
                break
              #if i in train_indices and i not in ignore_lines:
              #if i in train_indices:
              if i == train_indices[0]:
                trl1.write(line)
                train_indices.pop(0)
              print("line", i, "out of train set          \r", end="")
        else:
          _, self.train_files_size = self.getSentenceLengths(self.train_files[n])
          print(self.train_files_size)
        print()

        test_indices = copy.deepcopy(test_indices_)
        if not os.path.exists(self.test_files[n]):
          f1.seek(0)
          with open(self.test_files[n], "w") as tel1:
            for j, line in enumerate(f1):
              if len(test_indices) == 0:
                break
              if j == test_indices[0]:
                tel1.write(line)
                test_indices.pop(0)
              print("line", j, "out of test set          \r", end="")
        else:
          if n == 0:
            _, self.test_files_size = self.getSentenceLengths(self.test_files[n])
        print()

        validate_indices = copy.deepcopy(validate_indices_)
        if not os.path.exists(self.validate_files[n]):
          f1.seek(0)
          with open(self.validate_files[n], "w") as val1:
            f1.seek(0)
            for k, line in enumerate(f1):
              if len(validate_indices) == 0:
                break
              if k == validate_indices[0]:
                val1.write(line)
                validate_indices.pop(0)
              print("line", k, "out of validate set          \r", end="")
        else:
          if n == 0:
            _, self.validate_files_size = self.getSentenceLengths(self.validate_files[n])
        print()

    print("num train sentences: ", self.train_files_size)
    print("num test sentences: ", self.test_files_size)
    print("num validate sentences: ", self.validate_files_size)

  def tokensToWords(self, tokens, dictionary, no_unk=True):
    '''
    Expects an array of tokens of the form:
    [token0, token1, token2, token3, ...]
    where tokens are in numerical format.
    Returns a string of words delimited by spaces.
    '''
    words = []
    for token in tokens:
      if no_unk:
        if dictionary[token] != "<unk>":
          words.append(dictionary[token])
      else:
        words.append(dictionary[token])

    return " ".join(words)

  def oneHotsToTokens(self, one_hots):
    '''
    Expects a matrix, or a vector of one-hot vectors.
    '''

    tokens = []
    for i in range(len(one_hots)):
      tokens.append(one_hots[i].argmax() + 1)

    return tokens

  def oneHotsToWords(self, one_hots, dictionary, no_unk=True):
    '''
    Expects one_hots to be a numpy array with rows of one-hot values and
    columns that correspond to word IDs.
    Or one_hots can also be a list of one-hot vectors.
    '''
    tokens = []
    for oh in one_hots:
      #print(np.nonzero(oh))
      if not np.any(oh):
        continue
      token = np.squeeze(oh.argmax()) + 1
      tokens.append(int(token))
    return self.tokensToWords(tokens, dictionary, no_unk)

  def softmaxesToWords(self, softmaxes, dictionary, no_unk=True):
    '''
    Expects softmaxes to be a numpy array with normalized rows. This will take
    the argmax of each row, translate that into a token, and convert the
    accumulated tokens to a series of words.
    '''
    tokens = []
    for sm in softmaxes:
      token = sm.argmax() + 1 # Softmax indices are 0-indexed, tokens are 1-indexed.
      tokens.append(token)
    return self.tokensToWords(tokens, dictionary, no_unk=no_unk)

  def topKPredictions(self, softmaxes, k, dictionary):
    '''
    Expects softmaxes to be a numpy array with rows whose values are
    normalized.
    softmaxes: [[0.8, 0.1, 0.1], [0.5, 0.3, 0.2], ...]
    k: some int
    dictionary: {token1:word1, token2:word2, ...}
    '''
    topKTokens = []
    topKProbs = []
    for sm in softmaxes:
      rearranged = [(i, p) for i, p in enumerate(sm)]
      rearranged = sorted(rearranged, key=(lambda s: s[1]))[::-1]
      topKWordTokens = [t[0] + 1 for t in rearranged[:k]]
      topKProbs.append(tuple([t[1] for t in rearranged[:k]]))
      topKTokens.append(tuple(topKWordTokens))
    topKWords = [self.tokensToWords(t, dictionary, no_unk=False) for t in topKTokens]
    topKItems = [(w, p) for w, p in zip(topKWords, topKProbs)]
    return topKItems

  def getTrainBatch(self, batch_size):
    return self.getBatch(batch_size, source="train")

  def getTestBatch(self, batch_size):
    return self.getBatch(batch_size, source="test")

  def getValidateBatch(self, batch_size):
    return self.getBatch(batch_size, source="validate")

  def rawTextToOneHots(self, lines, vocab, seq_length=None, use_start=True):
    '''
    Expects lines to be an array of strings, with token characters delimited by
    spaces. So lines[i].split(" ") provides token numbers of the words in the
    line.
    Expects vocab to be a dictionary of words to token numbers. It also expects
    that the <end> tag is the last token in the dictionary (numerically).
    '''

    max_line_length = 0
    if seq_length == None:
      max_line_length = max([len(l.split(" ")) for l in lines]) + 2
    else:
      max_line_length = seq_length + 2
    batch_size = len(lines)
    vocab_size = len(vocab)
    one_hots = np.zeros((batch_size, max_line_length, vocab_size))
    #one_hots[:, 1:, vocab_size - 2] = 1.0 # Janky way to end-pad with "<end>" vector
    token_position_shift = 0
    if use_start:
      one_hots[:, 0, vocab_size - 1] = 1.0 # Janky way to front-pad with "<start>" vectors
      token_position_shift = 1

    for bs in range(batch_size):
      line_tokens = lines[bs].strip().split(" ")
      #print("line tokens: ", lines[bs], line_tokens)
      for sl in range(len(line_tokens)):
        hot_index = int(line_tokens[sl]) - 1 # Tokens are 1-indexed
        #one_hots[bs, sl + token_position_shift, vocab_size - 2] = 0.0
        one_hots[bs, sl + token_position_shift, hot_index] = 1.0
      if use_start:
        one_hots[bs, len(line_tokens) + 1, vocab_size - 2] = 1.0
      else:
        one_hots[bs, len(line_tokens), vocab_size - 2] = 1.0

    return one_hots

  def getBatch(self, batch_size, source):
    '''
    Return a batch of items of size 'batch_size' from the designated source.
    This method loads multiple batches' worth of data at a time. If the size
    of the preloaded data exceeds the size of the requested batch, then batch
    items are pulled out of the preloaded data (saves on file read/writes).
    '''
    batch_lines = [[], []]
    batch = [None, None]
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
          remaining_indices = list(set(all_indices).difference(set(self.used_train_indices)))
          self.num_epochs_elapsed += 1

        np.random.shuffle(remaining_indices)
        preloaded_indices = sorted(remaining_indices[0:num_sequences])
        self.used_train_indices += preloaded_indices
        for i in range(2):
          num_lines_loaded = 0
          with open(self.train_files[i]) as traf:
            for j, line in enumerate(traf):
              if len(self.preloaded_train_data[i]) == num_sequences:
                #print("preloaded training data ", i, " has ", len(self.preloaded_train_data[i]), "things in it")
                break
              if j == preloaded_indices[num_lines_loaded] and (len(line.split(" ")) > 0):
                print("indices remaining: ", num_sequences - len(self.preloaded_train_data[i]), "          \r", end="")
                self.preloaded_train_data[i].append(line)
                num_lines_loaded += 1
          #print("Finished loading batch indices from file ", self.train_files[i])

        batch_lines[0] = self.preloaded_train_data[0][0:batch_size]
        batch_lines[1] = self.preloaded_train_data[1][0:batch_size]
        del self.preloaded_train_data[0][0:batch_size]
        del self.preloaded_train_data[1][0:batch_size]

    elif source == "test":

      if len(self.preloaded_test_data[0]) != len(self.preloaded_test_data[1]):
        print("The two preloaded data items do not have the same number of items")
        sys.exit(-1)

      all_indices = np.arange(self.test_files_size)
      if (len(self.preloaded_test_data[0]) >= batch_size) and \
         (len(self.preloaded_test_data[1]) >= batch_size):
        batch_lines[0]  = self.preloaded_test_data[0][0:batch_size]
        batch_lines[1]  = self.preloaded_test_data[1][0:batch_size]
        del self.preloaded_test_data[0][0:batch_size]
        del self.preloaded_test_data[1][0:batch_size]
      else:
        num_sequences = batch_size*self._preloaded_batch_depth
        remaining_indices = list(set(all_indices).difference(set(self.used_test_indices)))
        print("Number of remaining indices: ", len(remaining_indices))
        # If the number of remaining indices drops below the number of lines
        # that must be loaded, then the used testing indices is reset and we
        # start sampling from the entire testing file.
        if len(remaining_indices) < num_sequences:
          print("Ran through the entire testing set! Resetting the used testing indices.")
          print("    This batch will have indices sampled from the entire dataset.")
          self.used_test_indices.clear()
          remaining_indices = list(set(all_indices).difference(set(self.used_test_indices)))

        np.random.shuffle(remaining_indices)
        preloaded_indices = sorted(remaining_indices[0:num_sequences])
        self.used_test_indices += preloaded_indices
        for i in range(2):
          num_lines_loaded = 0
          with open(self.test_files[i]) as traf:
            for j, line in enumerate(traf):
              if len(self.preloaded_test_data[i]) == num_sequences:
                #print("preloaded testing data ", i, " has ", len(self.preloaded_test_data[i]), "things in it")
                break
              if j == preloaded_indices[num_lines_loaded] and (len(line.split(" ")) > 0):
                print("indices remaining: ", num_sequences - len(self.preloaded_test_data[i]), "          \r", end="")
                self.preloaded_test_data[i].append(line)
                num_lines_loaded += 1
          #print("Finished loading batch indices from file ", self.test_files[i])

        batch_lines[0] = self.preloaded_test_data[0][0:batch_size]
        batch_lines[1] = self.preloaded_test_data[1][0:batch_size]
        del self.preloaded_test_data[0][0:batch_size]
        del self.preloaded_test_data[1][0:batch_size]

    elif source == "validate":

      if len(self.preloaded_validate_data[0]) != len(self.preloaded_validate_data[1]):
        print("The two preloaded data items do not have the same number of items")
        sys.exit(-1)

      all_indices = np.arange(self.validate_files_size)
      if (len(self.preloaded_validate_data[0]) >= batch_size) and \
         (len(self.preloaded_validate_data[1]) >= batch_size):
        batch_lines[0]  = self.preloaded_validate_data[0][0:batch_size]
        batch_lines[1]  = self.preloaded_validate_data[1][0:batch_size]
        del self.preloaded_validate_data[0][0:batch_size]
        del self.preloaded_validate_data[1][0:batch_size]
      else:
        num_sequences = batch_size*self._preloaded_batch_depth
        remaining_indices = list(set(all_indices).difference(set(self.used_validate_indices)))
        print("Number of remaining indices: ", len(remaining_indices))
        # If the number of remaining indices drops below the number of lines
        # that must be loaded, then the used validating indices is reset and we
        # start sampling from the entire validating file.
        if len(remaining_indices) < num_sequences:
          print("Ran through the entire validating set! Resetting the used validating indices.")
          print("    This batch will have indices sampled from the entire dataset.")
          self.used_validate_indices.clear()
          remaining_indices = list(set(all_indices).difference(set(self.used_validate_indices)))

        np.random.shuffle(remaining_indices)
        preloaded_indices = sorted(remaining_indices[0:num_sequences])
        self.used_validate_indices += preloaded_indices
        for i in range(2):
          num_lines_loaded = 0
          with open(self.validate_files[i]) as traf:
            for j, line in enumerate(traf):
              if len(self.preloaded_validate_data[i]) == num_sequences:
                #print("preloaded validating data ", i, " has ", len(self.preloaded_validate_data[i]), "things in it")
                break
              if j == preloaded_indices[num_lines_loaded] and (len(line.split(" ")) > 0):
                print("indices remaining: ", num_sequences - len(self.preloaded_validate_data[i]), "          \r", end="")
                self.preloaded_validate_data[i].append(line)
                num_lines_loaded += 1
          #print("Finished loading batch indices from file ", self.validate_files[i])

        batch_lines[0] = self.preloaded_validate_data[0][0:batch_size]
        batch_lines[1] = self.preloaded_validate_data[1][0:batch_size]
        del self.preloaded_validate_data[0][0:batch_size]
        del self.preloaded_validate_data[1][0:batch_size]

    max_line_lengths = []
    max_line_lengths.append(max([len(l.split(" ")) for l in batch_lines[0]]) + 1)
    max_line_lengths.append(max([len(l.split(" ")) for l in batch_lines[1]]) + 1)

    batch_lengths = [(i, len(l.split(" "))) for i, l in enumerate(batch_lines[0])]
    length_sorted_batch_indices = sorted(batch_lengths, key=(lambda s: s[1]))
    #print("length sorted batch indices: ", length_sorted_batch_indices)
    batch_lines_sorted = [None, None]
    for i in range(len(batch_lines)):
      batch_lines_sorted[i] = [batch_lines[i][j[0]] for j in length_sorted_batch_indices]

    for i, bstring in enumerate(batch_lines_sorted):
      use_start = True
      if i == 0:
        use_start = self._input_has_start_token
      if i == 1:
        use_start = self._output_has_start_token
      batch[i] = self.rawTextToOneHots(bstring, self.dict_word_to_token_langs[i], max(max_line_lengths), use_start)
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

  parser.add_argument("-n", "--numpulls",
    required    = False,
    default     = 5,
    type        = int,
    help        = "The number of batches to be pulled out of the data. \
                   Default value is 5.")

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

  dh = DataHandler(args.language1, args.dict1, args.language2, args.dict2, False, False)
  draw_times = []
  start_time = 0
  end_time = 0
  for i in range(args.numpulls):
    start_time = datetime.datetime.now()
    batch1 = dh.getTrainBatch(64)
    end_time = datetime.datetime.now()
    batch2 = dh.getTestBatch(64)
    print(batch1[0].shape, batch1[1].shape)
    print()
    print(dh.oneHotsToWords(batch1[0][0], dh.dict_token_to_word_langs[0], no_unk=False))
    print(dh.oneHotsToTokens(batch1[0][0]))
    print(dh.oneHotsToWords(batch1[1][0], dh.dict_token_to_word_langs[1], no_unk=False))
    print(dh.oneHotsToTokens(batch1[1][0]))
    print()
    print(dh.oneHotsToWords(batch2[0][0], dh.dict_token_to_word_langs[0], no_unk=False))
    print(dh.oneHotsToTokens(batch2[0][0]))
    print(dh.oneHotsToWords(batch2[1][0], dh.dict_token_to_word_langs[1], no_unk=False))
    print(dh.oneHotsToTokens(batch2[1][0]))
    print("\nend loop\n")
    diff = end_time - start_time
    draw_times.append(diff.seconds)
    print("That batch took ", diff.seconds, " seconds to be drawn.")
    
  #plt.show()

if __name__ == "__main__":
  main(sys.argv)