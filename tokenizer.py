import argparse
import json
import nltk
from nltk.tokenize import word_tokenize
import os
import sys

def getMostCommonWords(wordFrequencies, k):
  '''
  Assumes that the dictionary contains 
  {"word": count, ...}
  '''

  words = [word for word in wordFrequencies]
  allWords = sorted(words, key=(lambda w: wordFrequencies[w]))[::-1]
  print("There are ", len(allWords), " words in the corpus.")
  print(allWords[0:100])
  #sys.exit()
  mostCommonWords = allWords[0:k]
  return mostCommonWords

def main(args):
  # Assumes that data is aligned by newline characters.

  parser = argparse.ArgumentParser(
    description = "Tokenize monolingual text and split target text into equal-\
                   sized sentences.")

  parser.add_argument("-l", "--language",
    help        = "The language of the target. English is assumed if left \
                   blank. Choice must be \'english\', \'french\', or \
                   \'spanish\' (without quotes)",
    choices     = ["english", "french", "spanish"],
    default     = "english",
    required    = False)

  parser.add_argument("-t", "--text",
    help        = "The location of the monolingual text.",
    required    = True)

  parser.add_argument("-o", "--output",
    help        = "The location of the parsed output.",
    required    = True)

  parser.add_argument("-k", "--topk",
    help        = "The top k words that will be included in the tokenization. \
                   Not required, default value is 30000.",
    default     = 30000,
    type        = int,
    required    = False)

  parser.add_argument("-d", "--tokendict",
    help        = "The dictionary containing the words-to-tokens mapping.",
    default     = None,
    type        = str,
    required    = False)

  args = parser.parse_args()

  if not os.path.exists(args.text):
    print("Could not find file named", args.text, "for parsing.")
    print("Exiting.")
    sys.exit(-1)

  if not os.path.exists(args.output):
    print("Could not find output path", args.output)
    print("This path will be created.")
    os.mkdir(args.output)
  else:
    print("Output path", args.output, "already exists.")
    print("New tokenizations will be placed here.")

  wordFreqs = {}
  topKTokens = {}
  if args.tokendict == None:
    with open(args.text, "r") as inputLines:
      for i, line in enumerate(inputLines):
        tokenizedLine = word_tokenize(line, args.language)
        for token in tokenizedLine:
          if token.lower() in wordFreqs:
            wordFreqs[token.lower()] += 1
          else:
            wordFreqs[token.lower()] = 1
        
    print("Got all lowercase word frequencies.")
    topKWords = getMostCommonWords(wordFreqs, args.topk)
    print("Got the topk words from the file.")
    print(topKWords[0:100])

    topKTokens = {w:v + 1 for v, w in enumerate(topKWords)}
    print("Converted the topk words into tokens.")

  else:
    with open(args.tokendict, "r") as td:
      topKTokens = json.load(td)

  with open(args.text, "r") as inputLines:
    with open(os.path.join(args.output, "tokenized." + args.language), "w") as outFile:
      for line in inputLines:
        lineTokens = word_tokenize(line, args.language)
        lineTokens = [t.lower() for t in lineTokens]
        tokenizedLine = [str(topKTokens[t]) if t in topKTokens else str(int(args.topk) + 1) for t in lineTokens]
        tokenizedLine = " ".join(tokenizedLine)
        outFile.write(tokenizedLine + "\n")

  print("Wrote the tokenized file back to the hard drive.")

  topKTokens["<unk>"] = int(args.topk + 1)
  topKTokens["<end>"] = int(args.topk + 2)
  topKTokens["<start>"]= int(args.topk + 3)
  wordIndexDict = open(os.path.join(args.output, "dictionary." + args.language), "w")
  topKTokensJson = json.dump(topKTokens, wordIndexDict)
  wordIndexDict.close()

  print("Generated the token:word dictionary.")

if __name__ == "__main__":
  main(sys.argv)