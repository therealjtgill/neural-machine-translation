import tensorflow as tf
import numpy as np
import os
import sys

class NMT(object):

  def __init__(self,
               in_vocab_size=30000,
               out_vocab_size=30000):

    embedding_size  = 640
    num_encoder_nodes     = 1024
    num_decoder_nodes     = 1024

    with tf.variable_scope("nmt"):
      self.input_data   = tf.placeholder(dtype=tf.float32, shape=[None, None, in_vocab_size])
      self.output_data  = tf.placeholder(dtype=tf.float32, shape=[None, None, out_vocab_size])

      # Sequence lengths between the encoder and decoder can be different.
      batch_size     = tf.cast(tf.shape(self.input_data)[0], tf.int32)
      seq_length_enc = tf.cast(tf.shape(self.input_data)[1], tf.int32)
      seq_length_dec = tf.cast(tf.shape(self.output_data)[0], tf.int32)

      self.zero_states_enc  = []
      self.zero_states_dec  = []

      # Placeholders for forward and backward states of encoder bidirectional LSTM.
      ph_enc_f  = tf.placeholder(dtype=tf.float32, shape=[None, num_encoder_nodes])
      ph_enc_b  = tf.placeholder(dtype=tf.float32, shape=[None, num_encoder_nodes])

      # Placeholders for forward and backward states of decoder bidirectional LSTM.
      ph_dec_f  = tf.placeholder(dtype=tf.float32, shape=[None, num_encoder_nodes])
      ph_dec_b  = tf.placeholder(dtype=tf.float32, shape=[None, num_encoder_nodes])

      self.W_in_embed = tf.Variable(tf.random_normal((in_vocab_size, embedding_size),
                                                         stddev=0.01))
      self.b_in_embed = tf.Variable(tf.random_normal((embedding_size,),
                                                         stddev=0.01))

      input_2d = tf.reshape(self.input_data, [-1, in_vocab_size])
      embedded_input_2d = tf.nn.relu(tf.matmul(input_2d, self.W_in_embed) + self.b_in_embed)
      embedded_input = tf.reshape(embedded_input_2d, [batch_size, seq_length_enc, embedding_size])

      # Using GRUs because their outputs are the same as their hidden states,
      # which makes dynamic unrolling possible.
      self.gru_enc_f = tf.nn.rnn_cell.GRUCell(num_encoder_nodes)
      self.gru_enc_b = tf.nn.rnn_cell.GRUCell(num_encoder_nodes)

      gru_enc_out, gru_enc_state = \
        tf.nn.bidirectional_dynamic_rnn(self.gru_enc_f,
                                        self.gru_enc_b,
                                        embedded_input,
                                        initial_state_fw=ph_enc_f,
                                        initial_state_bw=ph_enc_b,
                                        dtype=tf.float32)

      self.gru_dec = tf.nn.rnn_cell(GRUCell(num_decoder_nodes))

      gru_dec_out, gru_dec_state = \
        tf.nn.dynamic_rnn(self.gru_dec,
                          )


if __name__ == "__main__":
  n = NMT()