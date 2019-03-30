import matplotlib.pyplot as plt
import numpy as np
import sys

def plotAttentionMatrix(attention, save_dir, english_labels, target_labels, file_name="attentionmatrix"):
  num_english_words, num_target_words = attention.shape
  fig = plt.figure()
  ax = fig.subplots()
  
  print(english_labels)
  print(target_labels)
  #plt.title("Attention matrix")
  plt.yticks(range(num_english_words), english_labels, fontsize=6)
  plt.xticks(range(num_target_words), target_labels, fontsize=6, rotation=90)
  plt.ylabel("English words")
  plt.xlabel("Target words")
  ax.xaxis.tick_top()
  plt.imshow(attention, interpolation="nearest", aspect=1)
  plt.show()

def main(argv):
  attention = np.random.rand(5, 6)
  english_words = "this is the best thing".split()
  french_words = "this is the best french thing".split()
  plotAttentionMatrix(attention, ".", english_words, french_words)

if __name__ == "__main__":
  main(sys.argv)