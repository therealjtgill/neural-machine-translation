from __future__ import print_function

import numpy as np
from numpy.random import rand

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import init_ops
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
  def __init__(self, input_size, gru_size, encoder_output, teacher_forcing=tf.constant(False, dtype=tf.bool), output_vocab_size=30000, output_embedding_size=620):
    print("encoder output: ", encoder_output)

    self._teacher_forcing = teacher_forcing

    assert((input_size % 2) == 0)
    self._input_size = input_size

    self._gru_size = gru_size
    self._in_seq_length = tf.cast(tf.shape(encoder_output)[0], tf.int32)

    # The embedded output from the decoder and the attention vector are snuck in with the actual decoder state.
    self._output_embedding_size = output_embedding_size
    self._output_vocab_size = output_vocab_size
    self._output_size = (self._output_vocab_size, self._in_seq_length)
    self._state_size = (self._gru_size, self._output_vocab_size, self._in_seq_length)

    # Shape = [batch_size, in_seq_length, input_size]
    self._encoder_output = encoder_output

    self._gru_cell = tf.nn.rnn_cell.GRUCell(self._gru_size, kernel_initializer=tf.initializers.orthogonal(gain=1.0, dtype=tf.float32))
    attention_size = 1024

    # Will be multiplied by input state.
    self.W_a  = vs.get_variable(
      "W_a",
      shape=[self._gru_size, attention_size],
      initializer=init_ops.random_normal_initializer(stddev=0.001))
    self.bw_a = vs.get_variable(
      "bw_a",
      shape=[attention_size],
      initializer=init_ops.random_normal_initializer(stddev=0.001))

    # Will be multiplied by hidden state from encoder.
    self.U_a  = vs.get_variable(
      "U_a",
      shape=[self._input_size, attention_size],
      initializer=init_ops.random_normal_initializer(stddev=0.001))
    self.bu_a = vs.get_variable(
      "bu_a",
      shape=[attention_size],
      initializer=init_ops.random_normal_initializer(stddev=0.001))

    # Used to get logits for attention mechanism.
    self.v_a  = vs.get_variable(
      "v_a",
      shape=[attention_size],
      initializer=init_ops.random_normal_initializer(stddev=0.01))

    # Embedding matrix.
    self.E    = vs.get_variable(
      "E",
      shape=[self._output_vocab_size, self._output_embedding_size],
      initializer=init_ops.random_normal_initializer(stddev=0.01))

    # Maxout #1 variables.
    self.U_p  = vs.get_variable(
      "U_p",
      shape=[self._gru_size, 500],
      initializer=init_ops.random_normal_initializer(stddev=0.01))
    self.V_p  = vs.get_variable(
      "V_p",
      shape=[self._output_embedding_size, 500],
      initializer=init_ops.random_normal_initializer(stddev=0.01))
    self.C_p  = vs.get_variable(
      "C_p",
      shape=[2*self._gru_size, 500],
      initializer=init_ops.random_normal_initializer(stddev=0.01))
    self.b_p  = vs.get_variable(
      "b_p",
      shape=[500],
      initializer=init_ops.random_normal_initializer(stddev=0.01))

    # Maxout #2 variables.
    self.U_q  = vs.get_variable(
      "U_q",
      shape=[self._gru_size, 500],
      initializer=init_ops.random_normal_initializer(stddev=0.01))
    self.V_q  = vs.get_variable(
      "V_q",
      shape=[self._output_embedding_size, 500],
      initializer=init_ops.random_normal_initializer(stddev=0.01))
    self.C_q  = vs.get_variable(
      "C_q",
      shape=[2*self._gru_size, 500],
      initializer=init_ops.random_normal_initializer(stddev=0.01))
    self.b_q  = vs.get_variable(
      "b_q",
      shape=[500],
      initializer=init_ops.random_normal_initializer(stddev=0.01))

    # Logits variables.
    self.W_o  = vs.get_variable(
      "W_o",
      shape=[500, self._output_vocab_size],
      initializer=init_ops.random_normal_initializer(stddev=0.01))
    self.bw_o = vs.get_variable(
      "bw_o",
      shape=[self._output_vocab_size],
      initializer=init_ops.random_normal_initializer(stddev=0.01))

    # Shape = [batch_size, in_seq_length, 1024]]
    self._precomputed = tf.einsum("bti,if->btf", self._encoder_output, self.U_a)

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
    The softmax/one-hot transformation and the attention output for each intermediate
    state is provided as the decoder's output.
    '''
    #return self._output_vocab_size
    return self._output_size

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
      state_true = state[0]

      print("state[1]: ", state[1])
      print("inputs: ", inputs)
      y_prev = math_ops.matmul(state[1], self.E)

      # Shape = [batch_size, in_seq_length, 512]]
      attention_d = tf.nn.tanh(tf.expand_dims(tf.matmul(state_true, self.W_a) + self.bw_a, axis=1) + self._precomputed)
      #attention_d = tf.nn.tanh(tf.expand_dims(tf.matmul(state_true, self.W_a), axis=1) + self._precomputed)
      print("attention_d: ", attention_d)
      # Shape = [batch_size, in_seq_length]
      attention_e = tf.einsum("i,bti->bt", self.v_a, attention_d)
      print("attention_e: ", attention_e)
      # Shape = [batch_size, in_seq_length]
      alpha = nn_ops.softmax(attention_e, axis=1)
      # Shape = [batch_size, input_size]
      context = tf.einsum("bt,bti->bi", alpha, self._encoder_output)
      print("gru state size: ", self._gru_cell.state_size)
      print("gru output size: ", self._gru_cell.output_size)
      print("array_ops.concat([context, y_prev]): ", array_ops.concat([context, y_prev], axis=-1))
      print("state_true: ", state_true)
      gru_out, gru_state = self._gru_cell(array_ops.concat([context, y_prev], axis=-1), state_true)

      p = math_ops.matmul(state_true, self.U_p) + math_ops.matmul(y_prev, self.V_p) + math_ops.matmul(context, self.C_p) + self.b_p
      #p = math_ops.matmul(state_true, self.U_p) + math_ops.matmul(y_prev, self.V_p) + math_ops.matmul(context, self.C_p)
      q = math_ops.matmul(state_true, self.U_q) + math_ops.matmul(y_prev, self.V_q) + math_ops.matmul(context, self.C_q) + self.b_q
      #q = math_ops.matmul(state_true, self.U_q) + math_ops.matmul(y_prev, self.V_q) + math_ops.matmul(context, self.C_q)

      print("p: ", p)
      print("q: ", q)

      t = gen_math_ops.maximum(p, q)
      out_logits = math_ops.matmul(t, self.W_o) + self.bw_o
      #out_logits = math_ops.matmul(t, self.W_o)

      output = (out_logits, alpha)

      f1 = lambda : inputs
      f2 = lambda : nn_ops.softmax(out_logits)
      state_softmax = tf.cond(self._teacher_forcing, f1, f2)

      state_out = (gru_state, state_softmax, alpha)
      print("gru state: ", gru_state)

      return output, state_out