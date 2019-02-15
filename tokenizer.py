import argparse
import json
import nltk
from nltk.tokenize import word_tokenize
import os
import sys

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

  textFile = open(args.text)
  allLines = textFile.readlines()
  #allLines = []
  #for i in range(1000):
  #  allLines.append(textFile.readline().replace("\n", ""))
  allSentences = " ".join(allLines)
  textFile.close()

  print("Opened the text file.")
  allWords = word_tokenize(allSentences)
  print("Loaded all sentences.")
  wordFreqs = nltk.FreqDist(w.lower() for w in allWords)
  print("Got all lowercase word frequencies.")
  topKWordFreqs = wordFreqs.most_common(int(args.topk))
  print("Got the topk words from the file.")
  topKWords = [w[0] for w in topKWordFreqs]
  print(topKWords[0:100])

  topKTokens = {w:v + 1 for v, w in enumerate(topKWords)}
  print("Converted the topk words into tokens.")
  print(topKTokens)
  
  with open(os.path.join(args.output, "tokenized." + args.language), "w") as outFile:
    for line in allLines:
      lineTokens = word_tokenize(line, args.language)
      lineTokens = [t.lower() for t in lineTokens]
      tokenizedLine = [str(topKTokens[t]) if t in topKTokens else str(int(args.topk) + 1) for t in lineTokens]
      tokenizedLine = " ".join(tokenizedLine)
      outFile.write(tokenizedLine + "\n")

  print("Wrote the tokenized file back to the hard drive.")

  topKTokens["<unk>"] = int(args.topk + 1)
  wordIndexDict = open(os.path.join(args.output, "dictionary." + args.language), "w")
  topKTokensJson = json.dump(topKTokens, wordIndexDict)
  #wordIndexDict.write(topKTokensJson)
  wordIndexDict.close()

  print("Generated the token:word dictionary.")

if __name__ == "__main__":
  main(sys.argv)