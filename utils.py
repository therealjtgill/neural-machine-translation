import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def saveAttentionMatrix(attention, save_dir, offset, english_labels, target_labels, file_name="attentionmatrix", suffix=""):
  num_target_words, num_english_words = attention.shape
  fig = plt.figure()
  ax = fig.subplots()
  
  #print(english_labels)
  #print(target_labels)
  #plt.title("Attention matrix")
  plt.xticks(range(num_english_words), english_labels, fontsize=6, rotation=90)
  plt.yticks(range(num_target_words), target_labels, fontsize=6)
  plt.xlabel("English words")
  plt.ylabel("Target words")
  ax.xaxis.tick_top()
  plt.imshow(attention, interpolation="nearest", aspect=1, cmap="gray", vmax=1.0, vmin=0.0)
  #plt.show()
  plt.savefig(os.path.join(save_dir, file_name + str(offset) + suffix + ".png"), dpi=600)
  plt.close()

def plotAttentionMatrix(attention, save_dir, english_labels, target_labels, file_name="attentionmatrix"):
  num_target_words, num_english_words = attention.shape
  fig = plt.figure()
  ax = fig.subplots()
  
  print(english_labels)
  print(target_labels)
  #plt.title("Attention matrix")
  plt.xticks(range(num_english_words), english_labels, fontsize=6, rotation=90)
  plt.yticks(range(num_target_words), target_labels, fontsize=6)
  plt.xlabel("English words")
  plt.ylabel("Target words")
  ax.xaxis.tick_top()
  plt.imshow(attention, interpolation="nearest", aspect=1, cmap="gray")
  plt.show()
  plt.close()

def main(argv):
  attention = np.random.rand(6, 5)
  english_words = "this is the best thing".split()
  french_words = "this is the best french thing".split()
  plotAttentionMatrix(attention, ".", english_words, french_words)

if __name__ == "__main__":
  main(sys.argv)