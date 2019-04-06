import numpy as np
import sys

class Node(object):
   '''
   This will keep track of a token value (corresponding to a word), and all
   necessary state information for the NN to make predictions.
   '''
   def __init__(self, vocab_size, initial_state):
      '''
      initial_state is a dictionary of the form
      {'decoder_state': None, 'decoder_out': None}
      '''
      self.__vocab_size = vocab_size
      self.initialize(None, None, 0., initial_state)

   def initialize(self, parent, action, value, state):
      self.__value = value
      self.__parent = parent
      self.__token = action
      self.__decoder_state = state['decoder_state']
      self.__decoder_out = state['decoder_out']

   @property
   def state(self):
      return {'decoder_state': self.__decoder_state,
      'decoder_out': self.__decoder_out}

   @property
   def parent(self):
      return self.__parent

   @property
   def value(self):
      return self.__value

   @property
   def token(self):
      '''
      A token is the hot-index + 1.
      '''
      return self.__token

   @property
   def prev_word(self):
      output = np.zeros((1, 1, self.__vocab_size))
      output[0, 0, self.__token - 1] = 1.
      return output

class Problem(object):
   def __init__(self, network, encoder_output):
      '''
      The Problem class will keep a reference to the neural network so that
      new nodes can be generated. The output of the encoder network is saved
      in the Problem class, because it will be used repeatedly.
      '''
      self.__network = network
      self.__encoder_output = encoder_output

   def actions(self, node):
      '''
      The contents of the node object will be given to the network to generate
      more possible tokens.
      '''

      # The encoder output needs to be saved as part of the Problem class.
      decoder_out, decoder_state, prediction = \
         self.__network.predictSingleStep(node.state["decoder_state"][0],
                                          node.prev_word,
                                          self.__encoder_output)
      k_hottest = self.__network.softmaxToKHottest(prediction)
      tokens = [t[0] + 1 for t in k_hottest]
      values = [v[1] for v in k_hottest]
      state = {
         "decoder_out": decoder_out,
         "decoder_state": decoder_state
      }
      return tokens, values, state

class BeamSearch(object):
   '''
   Needs to keep track of the initial states that come out of the network. Also
   needs to receive a reference to the network in its constructor.
   '''
   def __init__(self, problem, network, k_max=5, vocab_size=30000):
      self.__frontier = []
      self.__incumbent = None
      self.__problem = problem
      self.__k = k_max # Branching factor.
      self.__current_node = self.__problem.initialNode()
      self.__network = network
      self.__frontier.append(self.__current_node)
      self.__vocab_size = vocab_size

   def step(self):
      '''
      Returns True if there are more items left on the frontier. Returns false
      if the frontier is exhausted.
      '''

      if len(self.__frontier) == 0:
         print("No items left on the frontier!")
         return False

      node = self.__frontier.pop(-1)

      node_is_goal = self.__problem.goalTest(node)
      # Values are log probabilities, and we don't know how long the target
      # solution is. Heuristically, the optimal cost for any set of tokens is
      # 0.0, so the estimated cost is the current cost + 0.

      if node_is_goal and (node.value > self.__incumbent.value):
         self.__incumbent = node
         return (len(self.__frontier) > 0)
      if (not node_is_goal) and (node.value > self.__incumbent.value):
         actions, values, state = self.__problem.actions(node)
         for action, value in zip(actions, values):
            child = Node(self.__vocab_size)
            child.initialize(node, action, value, state)
            self.__frontier.append(child)

def main(argv):
   pass

if __name__ == "__main__":
   main(sys.argv)