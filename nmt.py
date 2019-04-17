from decodercell import DecoderCell
import numpy as np
import os
import sys
import tensorflow as tf

class NMT(object):

  def __init__(self,
               session,
               in_vocab_size=30000,
               out_vocab_size=30000,
               save=True,
               train=True):
    '''
    Set up the computation graph.
    '''
    embedding_size        = 640
    num_encoder_nodes     = 1000
    num_decoder_nodes     = 1000
    self.in_vocab_size = in_vocab_size
    self.out_vocab_size = out_vocab_size

    with tf.variable_scope("nmt", reuse=False):
      self.saver = None
      self.input_data_ph   = tf.placeholder(dtype=tf.float32, shape=[None, None, in_vocab_size], name="inputdata")
      self.output_data_ph  = tf.placeholder(dtype=tf.float32, shape=[None, None, out_vocab_size], name="outputdata")
      self.dropout_prob_ph = tf.placeholder(dtype=tf.float32, shape=[], name="dropoutprob")
      self.teacher_forcing_ph = tf.placeholder(dtype=tf.bool, shape=[], name="teacherforce")

      self.session = session

      # Sequence lengths between the encoder and decoder can be different.
      batch_size     = tf.cast(tf.shape(self.input_data_ph)[0], tf.int32)
      seq_length_enc = tf.cast(tf.shape(self.input_data_ph)[1], tf.int32)
      seq_length_dec = tf.cast(tf.shape(self.output_data_ph)[1], tf.int32)

      # Placeholders for forward and backward states of decoder bidirectional GRU.
      decoder_h_ph = tf.placeholder(dtype=tf.float32, shape=[None, num_encoder_nodes])

      self.W_in_embed  = tf.get_variable(
        "W_in_embed",
        shape=(in_vocab_size, embedding_size),
        initializer=tf.initializers.random_normal(stddev=0.01))
      self.bw_in_embed = tf.get_variable(
        "bw_in_embed",
        shape=(embedding_size,),
        initializer=tf.initializers.random_normal(stddev=0.01))

      input_2d = tf.reshape(self.input_data_ph, [-1, in_vocab_size])
      embedded_input_2d = tf.nn.relu(tf.matmul(input_2d, self.W_in_embed) + self.bw_in_embed)
      #embedded_input_2d = tf.matmul(input_2d, self.W_in_embed)
      embedded_input_3d = tf.reshape(embedded_input_2d, [batch_size, seq_length_enc, embedding_size])

      # Using GRUs because their outputs are the same as their hidden states,
      # which makes grabbing all unrolled hidden states possible.
      self.gru_encoder_fw = tf.nn.rnn_cell.GRUCell(
        num_encoder_nodes,
        kernel_initializer=tf.initializers.orthogonal(gain=1.0, dtype=tf.float32))
      self.gru_encoder_bw = tf.nn.rnn_cell.GRUCell(
        num_encoder_nodes,
        kernel_initializer=tf.initializers.orthogonal(gain=1.0, dtype=tf.float32))
      self.gru_encoder_fw_dropout = tf.nn.rnn_cell.DropoutWrapper(
        self.gru_encoder_fw,
        output_keep_prob=self.dropout_prob_ph)
      self.gru_encoder_bw_dropout = tf.nn.rnn_cell.DropoutWrapper(
        self.gru_encoder_bw,
        output_keep_prob=self.dropout_prob_ph)

      self.gru_encoder_out, self.gru_encoder_state = \
        tf.nn.bidirectional_dynamic_rnn(
          self.gru_encoder_fw_dropout,
          self.gru_encoder_bw_dropout,
          embedded_input_3d,
          dtype=tf.float32)

      W_decoder_init = tf.get_variable(
        "W_decoder_init",
        shape=(num_encoder_nodes, num_encoder_nodes),
        initializer=tf.initializers.random_normal())
      print("gru encoder out: ", self.gru_encoder_out)
      decoder_initial_gru_state = \
        tf.nn.tanh(tf.matmul(self.gru_encoder_out[1][:, 0, :], W_decoder_init))
      print("decoder initial state: ", decoder_initial_gru_state)
      self.gru_encoder_states = tf.concat(self.gru_encoder_out, axis=-1)

      self.gru_dec = DecoderCell(
        num_encoder_nodes*2,
        num_decoder_nodes,
        self.gru_encoder_states,
        teacher_forcing=self.teacher_forcing_ph,
        output_vocab_size=out_vocab_size)
      self.gru_dec_dropout = tf.nn.rnn_cell.DropoutWrapper(
        self.gru_dec,
        output_keep_prob=self.dropout_prob_ph)
      self.decoder_initial_state = \
        list(self.gru_dec_dropout.zero_state(batch_size, dtype=tf.float32))
      print("decoder state: ", self.decoder_initial_state)
      self.decoder_initial_state[0] = decoder_initial_gru_state
      self.decoder_initial_state = tuple(self.decoder_initial_state)
      print("state size decoder: ", self.gru_dec_dropout.state_size)

      # The decoder output for a single timestep is a tuple of:
      #   (softmax over target vocabulary, attention to input)
      self.gru_decoder_out, self.gru_decoder_state = \
        tf.nn.dynamic_rnn(
          self.gru_dec_dropout,
          self.output_data_ph,
          dtype=tf.float32,
          initial_state=self.decoder_initial_state)

      print("gru decoder out: ", self.gru_decoder_out)
      predicted_logits = self.gru_decoder_out[0]

      target_probs_flat = tf.reshape(
        self.output_data_ph,
        shape=[-1, out_vocab_size])

      predicted_logits_flat = tf.reshape(
        predicted_logits,
        shape=[-1, out_vocab_size])

      self.predictions = tf.nn.softmax(predicted_logits, axis=-1)
      self.attention   = self.gru_decoder_out[1]

      cross_entropy_2d = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_probs_flat, logits=predicted_logits_flat)
      cross_entropy_3d = tf.reshape(cross_entropy_2d, shape=[batch_size, seq_length_dec, -1])
      print("cross entropy: ", cross_entropy_3d)
      self.loss = tf.reduce_mean(tf.reduce_sum(cross_entropy_3d, axis=1))
      print("self loss: ", self.loss)

      if train:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, epsilon=1e-06)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_grads = [(grad if grad is None else tf.clip_by_norm(grad, 1.0), var) for grad, var in grads_and_vars]
        self.train_op = optimizer.apply_gradients(capped_grads)

      if save:
        self.saver = tf.train.Saver(max_to_keep=5)

    with tf.variable_scope("nmt", reuse=True):
      print(self.gru_dec_dropout.state_size)
      # The last element of the decoder's state is not used by the decoder, it's just for making pretty attention plots.
      self.decoder_gru_input_ph            = tf.placeholder(dtype=tf.float32, shape=(1, self.gru_dec_dropout.state_size[0]))
      self.decoder_prev_softmaxes_input_ph = tf.placeholder(dtype=tf.float32, shape=(1, self.gru_dec_dropout.state_size[1]))
      self.encoder_output_ph               = tf.placeholder(dtype=tf.float32, shape=[1, None, 2*num_encoder_nodes])

      fake_decoder_inputs = tf.zeros([1, 1, self.gru_dec_dropout.state_size[1]])

      self.gru_decoder_test = DecoderCell(
        num_encoder_nodes*2,
        num_decoder_nodes,
        self.encoder_output_ph,
        teacher_forcing=self.teacher_forcing_ph,
        output_vocab_size=out_vocab_size)

      decoder_input_state    = list(self.gru_decoder_test.zero_state(1, dtype=tf.float32))
      decoder_input_state[0] = self.decoder_gru_input_ph
      decoder_input_state[1] = self.decoder_prev_softmaxes_input_ph
      decoder_input_state    = tuple(decoder_input_state)
      self.gru_decoder_test_out, self.gru_decoder_test_state = \
        tf.nn.dynamic_rnn(
          self.gru_decoder_test,
          fake_decoder_inputs,
          dtype=tf.float32,
          initial_state=decoder_input_state)

      print("gru decoder test out: ", self.gru_decoder_test_out)
      predicted_test_logits = self.gru_decoder_test_out[0]

      self.predictions_test = tf.nn.softmax(predicted_test_logits, axis=-1)
      self.attention_test   = self.gru_decoder_test_out[1]

  def trainStep(self, X, y, dropout_keep_prob=0.8, teacher_forcing=True):
    '''
    Takes input and target output and performs one gradient update to the
    network weights.
    shape(X) = [batch_size, in_seq_length, in_vocab_size]
    shape(y) = [batch_size, out_seq_length, out_vocab_size]
    '''

    fetches = [
      self.loss,
      self.attention,
      self.train_op
    ]

    feeds = {
      self.input_data_ph   : X,
      self.output_data_ph  : y,
      self.dropout_prob_ph : dropout_keep_prob,
      self.teacher_forcing_ph : teacher_forcing
    }

    loss, attention, _ = self.session.run(fetches, feeds)

    return loss, attention

  def testStep(self, X, y):
    '''
    shape(X) = [batch_size, in_seq_length, in_vocab_size]
    shape(y) = [batch_size, out_seq_length, out_vocab_size]
    '''

    fetches = [
      self.loss,
      self.predictions,
      self.attention
    ]

    feeds = {
      self.input_data_ph   : X,
      self.output_data_ph  : y,
      self.dropout_prob_ph : 1.0,
      self.teacher_forcing_ph : False
    }

    loss, predictions, attention = self.session.run(fetches, feeds)

    return loss, predictions, attention

  def predict(self, X):
    '''
    shape(X) = [batch_size, in_seq_length, in_vocab_size]
    '''

    fetches = [
      self.predictions,
      self.attention
    ]

    feeds = {
      self.input_data_ph      : X,
      self.dropout_prob_ph    : 1.0,
      self.output_data_ph     : np.zeros_like(X),
      self.teacher_forcing_ph : False
    }

    predictions, attention = self.session.run(fetches, feeds)

    return predictions, attention

  def getEncoderOutputAndDecoderInput(self, X):
    '''
    shape(X) = [1, in_seq_length, in_vocab_size]
    '''

    fetches = [
      self.gru_encoder_out,
      self.decoder_initial_state
    ]

    feeds = {
      self.input_data_ph      : X,
      self.dropout_prob_ph    : 1.0,
      self.teacher_forcing_ph : False
    }

    encoder_out, decoder_init_state = self.session.run(fetches, feeds)

    return encoder_out, decoder_init_state

  def predictSingleStep(self, decoder_input, prev_word, encoder_output):
    '''
    shape(decoder_input) = [1, gru_size]
    shape(prev_word) = [1, output_vocab_size]
    shape(encoder_output) = [1, in_seq_length, 2*gru_size]
    '''
    
    fetches = [
      self.gru_decoder_test_out,
      self.gru_decoder_test_state,
      self.predictions_test,
      self.attention_test
    ]

    feeds = {
      self.decoder_gru_input_ph : decoder_input,
      self.decoder_prev_softmaxes_input_ph : np.reshape(prev_word, [1, -1]),
      self.encoder_output_ph : encoder_output,
      self.teacher_forcing_ph  : False
    }

    #print("decoder input shape: ", decoder_input.shape)
    #print("prev word shape: ", prev_word.shape)
    #print("encoder output shape: ", encoder_output.shape)

    #print(decoder_input[0].shape)
    #print(prev_word.shape)
    #print(encoder_output[0].shape)

    decoder_out, decoder_state, prediction, attention = self.session.run(fetches, feeds)
    #print("prediction shape: ", prediction.shape)
    return decoder_out, decoder_state, prediction, attention

  def predictSingleStepTopK(self, decoder_input, prev_word, encoder_output):
    decoder_out, decoder_state, prediction = \
      self.predictSingleStep(decoder_input, prev_word, encoder_output)

    top_k_items = self.softmaxToKHottest(self, prediction)

    return decoder_out, decoder_state, top_k_items

  def softmaxToKHottest(self, softmax, k=5):
    '''
    shape(softmax) = [1, 1, vocab_size]
    Returns a tuple of (index, conditional probability) for the top k predicted
    words.
    '''

    top_k_indices = []
    top_k_probs = []

    rearranged = [(i, p) for i, p in enumerate(softmax)]
    rearranged = sorted(rearranged, key=(lambda s: s[1]))[::-1]
    top_k_probs = tuple([t[1] for t in rearranged[:k]])
    top_k_indices = tuple([t[0] for t in rearranged[:k]])
    top_k_items = [(w, p) for w, p in zip(top_k_indices, top_k_probs)]
    return top_k_items

  def softmaxToOnehot(self, softmax):
    '''
    shape(softmax) = [1, 1, vocab_size]
    '''

    one_hot = np.zeros_like(softmax)
    no_unk_softmax = softmax
    no_unk_softmax[0, 0, 30000] = 0.0
    #hot_index = no_unk_softmax[0, 0].argmax()
    hot_index = softmax[0, 0].argmax()
    one_hot[0, 0, hot_index] = 1.0

    return one_hot, hot_index

  def greedySearch(self, X, stop_token=30001):

    encoder_out, decoder_init_state = self.getEncoderOutputAndDecoderInput(X)

    decoder_out, decoder_state, prediction, attention = \
      self.predictSingleStep(
        decoder_init_state[0],
        np.zeros((1, self.out_vocab_size)),
        np.concatenate(encoder_out, axis=-1)
        )

    one_hot, hot_index = self.softmaxToOnehot(prediction)
    hot_indices = []
    hot_indices.append(hot_index)
    attentions = []
    attentions.append(attention)

    while stop_token not in hot_indices:
      decoder_out, decoder_state, prediction, attention = \
        self.predictSingleStep(
          decoder_state[0],
          one_hot,
          np.concatenate(encoder_out, axis=-1)
          )

      one_hot, hot_index = self.softmaxToOnehot(prediction)
      hot_indices.append(hot_index)
      attentions.append(attention)

    print("attention shape: ", attentions[0].shape)

    return hot_indices, np.squeeze(np.concatenate(attentions, axis=1))

  def saveParams(self, save_dir, global_step):
    '''
    Save the model parameters.
    '''
    self.saver.save(self.session, save_dir, global_step=global_step)

  def loadParams(self, save_dir):
    '''
    Load saved model parameters.
    '''
    self.saver.restore(self.session, save_dir)

if __name__ == "__main__":
  '''
  Pass some fake data through the model as a water-through-pipes test.
  '''
  sess = tf.Session()
  input_batch = np.random.rand(20, 113, 30000)
  output_batch = np.random.rand(20, 113, 30000)
  n = NMT(sess)
  sess.run(tf.global_variables_initializer())
  print(n.trainStep(input_batch, output_batch)[0])
  print(n.testStep(input_batch, output_batch)[0])
  input_batch = np.random.rand(22, 56, 30000)
  print(n.predict(input_batch)[0])