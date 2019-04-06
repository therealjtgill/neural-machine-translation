import numpy as np

class Node(object):
   '''
   This will keep track of a token value (corresponding to a word), and all
   necessary state information for the NN to make predictions.
   '''
   def __init__(self):
      self.initialize()

   def initialize(self):
      self.__value = 0.0
      self.__parent = None
      self.__token = None
      self.__state = {}

   @property
   def state(self):
      return self.__state

   @property
   def parent(self):
      return self.__parent

   @property
   def value(self):
      return self.__value

   @property
   def token(self):
      return self.__token

class Problem(object):
   def __init__(self, network):
      '''
      The Problem class will keep a reference to the neural network so that
      new nodes can be generated.
      '''
      self.__network = network

   def actions(self, node):
      '''
      The contents of the node object will be given to the network to generate
      more possible tokens.
      '''
      

class BeamSearch(object):
   '''
   Needs to keep track of the initial states that come out of the network. Also
   needs to receive a reference to the network in its constructor.
   '''
   def __init__(self, problem, k_max, network):
      self.__frontier = []
      self.__incumbent = None
      self.__problem = problem
      self.__k = k_max # Branching factor.
      self.__current_node = None
      self.__network = network

   def step(self):
      if len(frontier) == 0:
         print("No items left on the frontier!")
         return

      current_node = self.__frontier.pop(-1)

      next_nodes = self.__problem.actions(current_node)

      for next_node in next_nodes:
         self.__frontier.append(next_node)


if __name__ == "__main__":
   pass