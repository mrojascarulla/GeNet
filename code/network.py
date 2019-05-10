from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import util

from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.layers import core as layers_core

import _pickle as pickle

class GENEt(object):
    """
    Neural network model for metagenomic classification. 
    """

    def __init__(self, iterator,
                 is_training,
                 config,
                 sigmoid_loss=False):

        #Overall training parameters
        self.connect_softmax = config.connect_softmax
        self.is_training = is_training
        self.batch_size = config.batch_size
        self.data_type = 'float32'
        self.mode = config.mode

        #Sequence params
        self.seq_length = config.seq_length
        self.num_labels = config.num_labels
        self.output_vocab = config.output_vocab
        self.vocab_size = config.vocab_size

        #Convnet encoder params 
        self.region_size = config.region_size
        self.num_filters = config.num_filters
        self.fully_connected = config.fully_connected
        self.num_fully_connected = config.num_fully_connected
        self.num_resnet_blocks = config.num_resnet_blocks
        self.weights = config.weights

        self.topk = config.topk

        # Learning rate params
        self.val_ac = []
        self.lr_patience = config.lr_patience
        self.lr_decay = config.lr_decay

        #Get new mini-batch
        (self.y, self.x,  self.y_path_abs, 
            self.y_levels) = iterator.get_next()

        self.x = tf.cast(self.x, dtype=tf.int32)
        self.x = tf.reshape(self.x, 
                            shape=[-1, self.seq_length])

        # Define embedding matrices for x and position
        embedding = tf.get_variable(
          "embedding", [self.vocab_size, self.vocab_size], 
          dtype=self.data_type)

        pos_embedding = tf.get_variable(
          "pos_embedding", [self.seq_length, self.vocab_size], 
          dtype=self.data_type)

        self.x_embed = tf.nn.embedding_lookup(embedding, self.x)
        positions = [tf.expand_dims(tf.range(self.x.shape[1]), 0) \
                      for r in range(self.batch_size)]
        positions = tf.concat(positions, 0)
        self.pos_embed = tf.nn.embedding_lookup(pos_embedding, positions)

        self.input = tf.one_hot(self.x, depth=self.vocab_size)
        self.input += self.x_embed + self.pos_embed
        self.input = tf.expand_dims(self.input, -1)

        # Build convolutional encoder of sequence
        self.build_conv_encoder()

        if self.mode == 'cnn':
            self.logits = tf.layers.dense(
              inputs=self.encoder_state,
              units=self.num_labels)

            self.loss = tf.losses.sparse_softmax_cross_entropy(
              labels=self.y,
              logits=self.logits)

            self.pred = tf.nn.top_k(self.logits, k=self.topk)[1]

        if self.mode == 'sg':
            num_units = config.num_groups
            num_groups = len(num_units)
            self.mb_weights = [self.weights[i] * tf.one_hot(tf.cast(self.y_levels[:, i], tf.int32), depth=num_units[i]) \
                       for i in range(num_groups)]

            self.mb_weights = [tf.reduce_sum(w, 1) for w in self.mb_weights]

            self.logits = [
              tf.layers.dense(self.encoder_state,
                               units=num_units[i],
                               activation=tf.nn.relu) 
               for i in range(num_groups)]

            if self.connect_softmax == 1:
                self.logits_add = [0] + [
                  tf.layers.dense(self.logits[i - 1], 
                                  units=num_units[i],
                                  activation=tf.nn.relu)
                    for i in range(1, num_groups)]
                self.logits = [
                  orig + new for (orig, new) in zip(self.logits, self.logits_add)]

            self.ce = [self.mb_weights[i] * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y_levels[:,i],
                    logits=self.logits[i]) 
                  for i in range(num_groups)]

            self.loss = tf.reduce_sum(self.ce) / self.batch_size
            self.pred = [tf.nn.top_k(self.logits[i], k=self.topk)[1]
                         for i in range(num_groups)]

        # Update parameters
        self.lr = tf.Variable(config.learning_rate, trainable=False)
        if self.is_training:

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss,params)
                optimizer = tf.train.MomentumOptimizer(self.lr, 0.9, use_nesterov=True)

            clipped_grads = gradients
            self.update_step = optimizer.apply_gradients(
              zip(clipped_grads,params))

            self.lr_new = tf.placeholder(tf.float32, shape=[], name='new_lr')
            self.lr_update = tf.assign(self.lr, self.lr_new)

        self.step = tf.Variable(1, trainable=False)
        self.new_step = tf.placeholder(tf.int32, shape=[], name='step')
        self.step_update = tf.assign(self.step, self.new_step)

    def resnet_block(self, input, region_size, conv_filter_shape, name):
      
        n_filters = conv_filter_shape[-1]
        n_filters_old = conv_filter_shape[-2]
        change_input_dim = conv_filter_shape[-1] != conv_filter_shape[-2]

        #Downsample when changing resnet block, divide size by 2
        if change_input_dim:
            input = tf.nn.pool(input, [2, 1], strides=[2, 1], pooling_type='AVG',
                                  padding='VALID')

        input_tr = tf.layers.batch_normalization(input, training=self.is_training)
        input_tr = tf.nn.relu(input_tr)


        conv1 = util.get_conv_layer(input_tr, strides=[1, 1, 1, 1],
                                    conv_filter_shape=conv_filter_shape,
                                    name=name, 
                                    padding='SAME')

        conv1 = tf.layers.batch_normalization(conv1, training=self.is_training)
        conv1 = tf.nn.relu(conv1)

        conv_filter_shape[-2] = conv_filter_shape[-1]
        conv2 = util.get_conv_layer(conv1, strides=[1, 1, 1, 1],
                                    conv_filter_shape=conv_filter_shape,
                                    name=name+'_2', padding='SAME')

        if change_input_dim:
            conv_project = util.get_conv_layer(input, [1, 1, 1, 1],
                                             [1, 1, n_filters_old, n_filters],
                                             'project',
                                             padding='SAME')
        else:
            conv_project = input
        
        return conv_project + conv2


    def build_conv_encoder(self):
        """
        Build convolutional encoder for sequence, 
        seq -> encoded_state
        """
        with tf.variable_scope("EncoderCNN") as vs:
          pooled = []
          n_filters = self.num_filters
          for region in [self.region_size]:
              self.conv = []
              current_input = self.input
              conv_filter_shape = [region, self.vocab_size, 1, n_filters]

              conv_1 = util.get_conv_layer(current_input, strides=[1, region, 1, 1], 
                                           conv_filter_shape=conv_filter_shape,
                                           name='conv_project')
              init_filters = n_filters
              conv_filter_resnet = [region, 1, init_filters, init_filters]

              conv = self.resnet_block(conv_1, region, conv_filter_resnet, 
                                       name='conv1')

              conv = self.resnet_block(conv, region, conv_filter_resnet, 
                                        name='conv2')


              for l in range(self.num_resnet_blocks):
                  conv_filter_resnet = [region, 1, init_filters, 2 * init_filters]

                  conv = self.resnet_block(conv, region, conv_filter_resnet, 
                                            name='conv'+str(l))

                  conv_filter_resnet = [region, 1, 2 * init_filters, 2 * init_filters]
                  conv = self.resnet_block(conv, region, conv_filter_resnet,
                                            name='conv'+str(l))

                  init_filters = 2 * init_filters

              # Average pooling
              self.conv = conv 
              conv = tf.layers.batch_normalization(conv, training=self.is_training)
              conv = tf.nn.relu(conv)
              mp = tf.reduce_mean(conv, [1, 2])
              pooled.append(mp)

          self.pooled = pooled
          self.encoder_state = tf.concat(pooled, 1)

          self.encoder_state = tf.layers.batch_normalization(
            self.encoder_state,
            training=self.is_training)

          self.encoder_state = tf.layers.dense(
              inputs=self.encoder_state,
              units=self.fully_connected,
              activation=tf.nn.relu)

