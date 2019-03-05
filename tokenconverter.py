import json

def tokensToOneHots(tokens, vocab_size):
  '''
  Expects an array of tokens of the form:
  [token0, token1, token2, token3, ...]
  The vocabulary size must be provided to ensure that the one-hot vectors have
  the correct length.
  '''
  one_hots = np.zeros((len(tokens), vocab_size))
  for i, t in enumerate(tokens):
    one_hots[i, t] = 1.0
  return one_hots

def tokensToWords(tokens, dictionary, no_unk=True):
  '''
  Expects an array of tokens of the form:
  [token0, token1, token2, token3, ...]
  where tokens are in numerical format.
  Returns a string of words delimited by spaces.
  '''
  words = []
  for token in tokens:
    #print(token)
    if no_unk:
      if dictionary[token] != "<unk>":
        words.append(dictionary[token])
    else:
      words.append(dictionary[token])

  return " ".join(words)

def oneHotsToWords(one_hots, dictionary, no_unk=True):
  '''
  Expects one_hots to be a numpy array with rows of one-hot values and
  columns that correspond to word IDs.
  Or one_hots can also be a list of one-hot vectors.
  '''
  tokens = []
  for oh in one_hots:
    #print(np.nonzero(oh))
    token = np.squeeze(oh.argmax())
    tokens.append(int(token))
  return tokensToWords(tokens, dictionary, no_unk)

def softmaxesToWords(softmaxes, dictionary, no_unk=True):
  '''
  Expects softmaxes to be a numpy array with normalized rows. This will take
  the argmax of each row, translate that into a token, and convert the
  accumulated tokens to a series of words.
  '''
  tokens = []
  for sm in softmaxes:
    token = sm.argmax()
    tokens.append(token)
  return tokensToWords(tokens, dictionary)

def topKPredictions(softmaxes, k, dictionary):
  '''
  Expects softmaxes to be a numpy array with rows whose values are
  normalized.
  softmaxes: [[0.8, 0.1, 0.1], [0.5, 0.3, 0.2], ...]
  k: some int
  dictionary: {token1:word1, token2:word2, ...}
  '''
  top_k_tokens = []
  for sm in softmaxes:
    rearranged = [(i, p) for i, p in enumerate(sm)]
    rearranged = sorted(rearranged, key=(lambda s: s[1]))
    top_word_k_tokens = [t[0] for t in rearranged[:k]]
    top_k_tokens.append(tuple(top_k))
  top_k_words = [tokensToWords(t, dictionary) for t in top_k_tokens]
  return top_k_words