from __future__ import print_function

import numpy as np
from numpy.random import rand

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.ops import random_ops
import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import RNNCell as RNNCell

class DecoderCell(RNNCell):
  '''
  The decoder is a wrapper around a GRUCell with an attention mechanism
  snuck in.
  This is an implementation of Bahdanau's decoder cell (2015).
  '''
  def __init__(self, input_size, gru_size, decoder_output, output_vocab_size=30000, output_embedding_size=620):
    self._input_size = input_size
    asset((self._input_size % 2) == 0)
    self._gru_size = gru_size
    # The embedded output from the decoder is snuck in with the actual decoder state.
    self._state_size = (self._input_size/2, output_embedding_size)
    self._output_embedding_size = output_embedding_size
    self._output_vocab_size = output_vocab_size
    # This thing needs to be dragged around to calculate attention at each call of
    # the decoder.
    # Shape = [batch_size, in_seq_length, input_size]
    self._decoder_output = decoder_output

    self._gru_cell = tf.nn.rnn_cell.GRUCell()

    # Will be multiplied by input state.
    self.W_a = variables.Variable(random_ops.random_normal(shape=[self._input_size/2, 512], stddev=0.001))
    self.bw_a = variables.Variable(random_ops.random_normal(shape=[self._input_size/2], stddev=0.001))

    # Will be multiplied by hidden state from encoder.
    self.U_a = variables.Variable(random_ops.random_normal(shape=[self._input_size, 512], stddev=0.001))
    self.bu_a = variables.Variable(random_ops.random_normal(shape=[self._input_size], stddev=0.001))

    # Dot producted 
    self.v_a = variables.Variable(random_ops.random_normal(shape=[512], stddev=0.001))

    self.E = variables.Variable(random_ops.random_normal(shape=[self._output_vocab_size, self._input_size], stddev=0.001))

    self.F = variables.Variable(random_ops.random_normal(shape=[self._output_embedding_size, self._output_vocab_size], stddev=0.001))

    # Shape = [batch_size, 512]
    self._precomputed = [tf.matmul(h, self.U_a) + self.bu_a for h in tf.split(self._decoder_output, axis=1)]

  @property
  def state_size(self):
  '''
  Contains the actual recurrent (hidden) state of the decoder and the decoder's
  embedded output (one-hot output passed through the decoder).
  '''
    return self._state_size

  @property
  def output_size(self):
  '''
  The softmax/one-hot transformation is provided as the decoder.
  '''
    return self._output_vocab_size

  def __call__(self, inputs, state, scope=None):
    '''
    Inputs have the shape: [batch_size, _input_size]
    State has the shape:   [batch_size, _state_size]
    The state_size is a tuple; the first item in state_size is the actual
    hidden state returned by the GRU, the second item is the embedded output
    of the target language.
    '''

    dtype = tf.float32

    with vs.variable_scope(scope or "decoder_cell"):
      # State smuggling
      true_state = state[0]
      y_prev = state[1]
      # Shape = [j, [batch_size, 512]]
      attentions_d = [tf.nn.tanh(tf.matmul(true_state, self.W_a) + self.bw_a + pc) for pc in self._precomputed]
      # Shape = [j, [batch_size, 1]]
      attentions_e = [tf.matmul(d, tf.expand_dims(self.v_a, 1)) for d in attentions_d]
      attentions_alpha_i = []

