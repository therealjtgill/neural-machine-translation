from decodercell import DecoderCell
import numpy as np
import os
import sys
import tensorflow as tf

class NMT(object):

  def __init__(self,
               session,
               in_vocab_size=30000,
               out_vocab_size=30000):

    embedding_size        = 640
    num_encoder_nodes     = 1024
    num_decoder_nodes     = 1024

    with tf.variable_scope("nmt"):
      self.input_data_ph   = tf.placeholder(dtype=tf.float32, shape=[None, None, in_vocab_size])
      self.output_data_ph  = tf.placeholder(dtype=tf.float32, shape=[None, None, out_vocab_size])
      self.session = session

      # Sequence lengths between the encoder and decoder can be different.
      batch_size     = tf.cast(tf.shape(self.input_data_ph)[0], tf.int32)
      seq_length_enc = tf.cast(tf.shape(self.input_data_ph)[1], tf.int32)
      seq_length_dec = tf.cast(tf.shape(self.output_data_ph)[0], tf.int32)

      self.zero_states_enc  = []
      self.zero_states_dec  = []

      # Placeholders for forward and backward states of encoder bidirectional GRU.
      encoder_h_fw_ph  = tf.placeholder(dtype=tf.float32, shape=[None, num_encoder_nodes])
      encoder_h_bw_ph  = tf.placeholder(dtype=tf.float32, shape=[None, num_encoder_nodes])

      # Placeholders for forward and backward states of decoder bidirectional GRU.
      decoder_h_ph = tf.placeholder(dtype=tf.float32, shape=[None, num_encoder_nodes])

      self.W_in_embed  = tf.Variable(tf.random_normal((in_vocab_size,
                                                       embedding_size),
                                                      stddev=0.01))
      self.bw_in_embed = tf.Variable(tf.random_normal((embedding_size,),
                                                      stddev=0.01))

      input_2d = tf.reshape(self.input_data_ph, [-1, in_vocab_size])
      embedded_input_2d = tf.nn.relu(tf.matmul(input_2d, self.W_in_embed) + self.bw_in_embed)
      embedded_input_3d = tf.reshape(embedded_input_2d, [batch_size, seq_length_enc, embedding_size])

      # Using GRUs because their outputs are the same as their hidden states,
      # which makes grabbing all unrolled hidden states possible.
      self.gru_encoder_fw = tf.nn.rnn_cell.GRUCell(num_encoder_nodes)
      self.gru_encoder_bw = tf.nn.rnn_cell.GRUCell(num_encoder_nodes)

      gru_encoder_out, gru_encoder_state = \
        tf.nn.bidirectional_dynamic_rnn(self.gru_encoder_fw,
                                        self.gru_encoder_bw,
                                        embedded_input_3d,
                                        initial_state_fw=encoder_h_fw_ph,
                                        initial_state_bw=encoder_h_bw_ph,
                                        dtype=tf.float32)

      gru_encoder_states = tf.concat(gru_encoder_out, axis=-1)
      self.gru_dec = DecoderCell(num_encoder_nodes*2, num_decoder_nodes, gru_encoder_states, output_vocab_size=out_vocab_size)

      # The decoder output for a single timestep is a tuple of:
      #   (softmax over target vocabulary, attention to input)
      gru_decoder_out, gru_decoder_state = \
        tf.nn.dynamic_rnn(self.gru_dec, embedded_input_3d, dtype=tf.float32)

      print("gru decoder out: ", gru_decoder_out)
      predicted_logits = gru_decoder_out[0]

      target_probs_flat     = tf.reshape(self.input_data_ph, shape=[-1, out_vocab_size])
      predicted_logits_flat = tf.reshape(predicted_logits, shape=[-1, out_vocab_size])
      self.predictions      = tf.nn.softmax(predicted_logits, axis=-1)
      self.attention        = gru_decoder_out[1]

      cross_entropy_2d = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_probs_flat, logits=predicted_logits_flat)
      cross_entropy_3d = tf.reshape(cross_entropy_2d, shape=[batch_size, seq_length_dec, -1])
      print("cross entropy: ", cross_entropy_3d)
      self.loss = tf.reduce_mean(tf.reduce_sum(cross_entropy_3d, axis=1))
      print("self loss: ", self.loss)

      optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, momentum=0.95)
      #grads_and_vars = optimizer.compute_gradients(self.loss)
      #capped_grads = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grads_and_vars]

      #self.train_op = optimizer.apply_gradients(capped_grads)
      self.train_op = optimizer.minimize(self.loss)

  def trainStep(self, X, y):
    '''
    Takes input and target output and performs one gradient update to the
    network weights.
    '''

    fetches = [
      self.loss,
      self.predictions,
      self.train_op
    ]

    feeds = {
      self.input_data_ph  : X,
      self.output_data_ph : y
    }

    loss, predictions, _ = self.session.run(fetches, feeds)

if __name__ == "__main__":
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  n = NMT(sess)