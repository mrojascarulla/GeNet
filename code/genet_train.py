from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import itertools
import tensorflow as tf
import network
import taxotree
import util
import IPython

from tensorflow.python.client import device_lib

from sklearn.preprocessing import OneHotEncoder
import _pickle as pickle
import sys
import os
import io

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("dir_path", None,
                    "Directory with genomes, dataset and taxonomy.")
flags.DEFINE_string("dataset_list", 'ncbi_ids.csv',
                    "Name of text file with list of the \
                    ncbi ids of the microbes in in the training set")
flags.DEFINE_string("save_path", '../saved_models',
                    "Directory to save checkpoint files")
flags.DEFINE_integer("num_gpus", 1,
                     "Number of available GPUs")
flags.DEFINE_integer("num_resnet_blocks", 1,
                     "Number of resnet blocks in encoder")
flags.DEFINE_integer("connect_softmax", 1,
                     "Whether to connect softmax act of previus group to next")
flags.DEFINE_integer("num_filters", 128,
                     "Number of filters in the first conv layer")
flags.DEFINE_integer("region_size", 3,
                     "Size of filter in the row dimension")
flags.DEFINE_integer("fully_connected", 1024,
                     "Fully connected units after conv")
flags.DEFINE_integer("batch_size", 64,
                     "Number of sequences in each mini batch")
flags.DEFINE_integer("read_length", 10000,
                     "Length of reads during training")
flags.DEFINE_float("lr_init", 1.,
                     "Initial learning rate for optimizer")
flags.DEFINE_integer("phred_score", 15,
                     "Quality score for sequencing, proba_accept = 10 ** (-ps/10)")
flags.DEFINE_string("mode", 'sg',
                    "Chosen architecture")
FLAGS = flags.FLAGS

class Config(object):
    init_scale = 0.1
    learning_rate = FLAGS.lr_init
    num_resnet_blocks = FLAGS.num_resnet_blocks
    num_filters = FLAGS.num_filters
    fully_connected = FLAGS.fully_connected
    num_fully_connected = 1
    connect_softmax = FLAGS.connect_softmax
    seq_length = FLAGS.read_length
    batch_size = FLAGS.batch_size
    region_size = FLAGS.region_size
    vocab_size = 6
    output_vocab = 100
    mode = FLAGS.mode
    topk = 1
    lr_threshold = 0.995
    lr_patience = 400
    lr_decay = 0.5

def main(_):
    gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
     ]

    # Load entire taxonomy.
    path_to_taxo = os.path.join(FLAGS.dir_path, 'nodes.dmp')
    path_to_dataset = os.path.join(FLAGS.dir_path, FLAGS.dataset_list)
    path_to_genomes = os.path.join(FLAGS.dir_path, 'genomes')


    if not os.path.exists(path_to_genomes):
        print("Downloading genomes...")
        util.download(path_to_dataset, path_to_genomes)
        print("Pickle genomes...")
        util.pickle_genome(path_to_genomes)

    print("Building taxonomy tree...")
    taxo_tree = taxotree.TaxoTree(path_to_taxo)

    # Keep only relevant parts of the tree
    taxo_tree.trim_to_dataset(path_to_dataset)
    # Load all the genomes to memory 
    taxo_tree.load_genomes(path_to_genomes)

    config=Config()
    num_groups = []
    for g in taxo_tree.groups:
        num_groups.append(taxo_tree.num_groups[g])

    num_groups.append(taxo_tree.num_labels)
    config.num_groups = num_groups
    config.num_labels = taxo_tree.num_labels
    config.weights = taxo_tree.proportions
    w_labels = [1./config.num_labels for i in range(config.num_labels)]
    config.weights.append(w_labels)

    print("Number of labels: %d" % taxo_tree.num_labels)

    # Define dataset generator
    genome_idx = np.arange(len(taxo_tree.genomes))
    def gen():
        for idx in genome_idx:
            yield (idx, taxo_tree.genomes[idx], taxo_tree.paths[idx], 
                   taxo_tree.paths_full[idx], taxo_tree.genus_species[idx])

    with tf.Graph().as_default():
        tf.set_random_seed(1234)
        
        proba_accept = 1 - 10 ** (-0.1 * int(FLAGS.phred_score))

        # Define dataset
        iterator = util.create_dataset_from_generator(gen, 
                                                      FLAGS.batch_size, 
                                                      proba_accept, 
                                                      FLAGS.read_length)
        # Define dataset for validation
        iterator_valid = util.create_dataset_from_generator(gen,
                                                            FLAGS.batch_size,
                                                            proba_accept,
                                                            FLAGS.read_length)

        initializer = tf.random_uniform_initializer(-0.1,
                                                    0.1)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, 
                                   initializer=initializer):
                N = network.GENEt(iterator,
                                      is_training=True,
                                      config=config)

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, 
                                   initializer=initializer):
                N_valid = network.GENEt(iterator_valid,
                                            is_training=False,
                                            config=config)

        #saver = tf.train.Saver(max_to_keep=5)
        num_p = (np.sum([np.prod(v.get_shape().as_list()) 
                 for v in tf.trainable_variables()]))
        logging.error("Number of parameters: %d" % (num_p))

        saver = tf.train.Saver(max_to_keep=3)
        if not os.path.exists(FLAGS.save_path):
            os.makedirs(FLAGS.save_path)

        with tf.Session() as session:
            
            summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.save_path, 
                                                                'tb_logs'))
            # Initialize variables.
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.save_path)
            if checkpoint_path:
                saver.restore(session, checkpoint_path)
            else:
                session.run(tf.global_variables_initializer())
                session.run(tf.local_variables_initializer())
            
            save_count = 0
            validation_ac = []
            lr_final = 0.1
            decay = np.exp(1./300 * np.log(lr_final / FLAGS.lr_init))

            for mb in itertools.count(N.step.eval()):
                ti = time.time()
                loss, _= session.run([N.loss, N.update_step])

                session.run(N.step_update, {N.new_step: mb})

                counter = 0
                if (mb - 1) % 10000 == 0 and mb > 1:
                    ti =time.time()
                    summary = tf.Summary()
                    summary.value.add(tag='Training loss (mini_batch)',
                                      simple_value=loss)
                    summary_writer.add_summary(summary, mb)

                    # Compute accuracy on validation (uniform error).
                    ac_val = util.compute_accuracy(session,
                                                   N_valid,
                                                   50,
                                                   config.mode, 
                                                   taxo_tree)
                    for k in ac_val[0]:
                        summary = tf.Summary()
                        summary.value.add(
                          tag='Val_accuracy_' + k + '_top_' + str(1),
                          simple_value=ac_val[0][k])
                        summary_writer.add_summary(summary, mb)

                    for k in ac_val[1]:
                        summary = tf.Summary()
                        summary.value.add(
                          tag='Val_accuracy_' + k + '_top_' + str(5),
                          simple_value=ac_val[1][k])
                        summary_writer.add_summary(summary, mb)

                    validation_ac.append(ac_val[0]['leaf'])
                    
                    if counter < 500:
                        lr_new = decay * N.lr.eval()
                        session.run(N.lr_update, {N.lr_new: lr_new}) 
                       
                    counter += 1
                    if lr_new != N.lr.eval():
                        validation_ac = []


                    summary = tf.Summary()
                    summary.value.add(tag='Learning_rate',
                                      simple_value=lr_new)
                    summary_writer.add_summary(summary, mb)

                    # Save checkpoint file
                    saver.save(session, os.path.join(FLAGS.save_path, 'model'), 
                               global_step=save_count)

                    save_count += 1
                
if __name__ == "__main__":
    tf.app.run()
