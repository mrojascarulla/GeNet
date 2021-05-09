from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import _pickle as pickle
import os
import io
import csv

from Bio import Entrez
from Bio import SeqIO

def download(dataset_csv_path, save_path):
    """
    Download genomes from NCBI's nucleotide database.

    Given the microbe ids given in a csv file, fetches the genomes
    corresponding to those ids from the nucleotide database.

    Args:
        dataset_csv_path: path to csv file containing the genome ids.
        save_path: save directory for the output FASTA files.

    Returns:
        None
    """

    Entrez.email = "your_email@gmail.com"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(dataset_csv_path, 'r') as f:
        data = csv.reader(f)
        for row in data:
            microbe_id = row[0].split('.')[0]
            if os.path.exists(os.path.join(save_path, microbe_id + '.fasta')):
                continue

            handle = Entrez.efetch(db="nucleotide", id=microbe_id,
                                   rettype="fasta", retmode="text")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            SeqIO.write(record, os.path.join(save_path, microbe_id + ".fasta"),
                        "fasta")

def pickle_genome(genomes_path):
    """
    Save genomes in FASTA files to pickle of bytes
    """

    word_to_id = {'A':0, 'T':1, 'C':2, 'G':3, 
                  'N':4, 'R':4, 'Y':4, 'K':4,
                  'M':4, 'S':4, 'W':4, 'B':4,
                  'D':4, 'H':4, 'V':4}

    for f in os.listdir(genomes_path):

        microbe = f.split('.')[0]
        if (f.split('.')[-1] != 'fasta' or 
            os.path.exists(os.path.join(genomes_path, microbe + '.pkl'))):
            continue

        for rec in SeqIO.parse(os.path.join(genomes_path, f), 'fasta'):
            seq = rec.seq
            with open(os.path.join(genomes_path, microbe + '.pkl'), 'wb') as g:
                pickle.dump(np.array([word_to_id[c] for c in seq], 
                                     dtype=np.uint8), g)

def read_words_from_array(l):

  word_to_id = {'A':0, 'T':1, 'C':2, 'G':3, 'N':4}
  return [word_to_id[c] for c in l]

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_to_tfr(x, y, writer):

    x_fwd = read_words_from_array(x[0])
    x_fwd = np.array(x_fwd, dtype=np.uint8)

    seq_fwd = x_fwd.tostring()
    lab = np.array(y).tostring()
    example = tf.train.Example(
            features=tf.train.Features(
              feature={'seq_fwd' : _bytes_feature(seq_fwd),
                       'label' : _bytes_feature(lab)}))
    writer.write(example.SerializeToString())

def tfr_record_size(path_to_tfr):
    c = 0
    for record in tf.python_io.tf_record_iterator(path_to_tfr):
        c += 1
    return c

# Define generator and parse function for tf dataset
def _parse_function_wrap(accept_proba, read_length):
    def _parse_function(idx, genome, path, path_full, levels,
                        read_length=read_length, 
                        accept_proba=accept_proba):

        """
        Create dataset on the fly by adding uniform noise to the reads. 
        """
        paddings = [[0, tf.reduce_max([read_length - tf.size(genome) + 1, 0])]]
        genome = tf.pad(genome, paddings, 'CONSTANT')
        length_genome = tf.size(genome)
        location = tf.random_uniform(maxval=length_genome - read_length, 
                                     shape=[1], dtype=tf.int32)

        read  = genome[location[0]:location[0] + read_length]
        where_to_flip = tf.random_uniform([read_length])
        where_to_flip = tf.cast(where_to_flip > accept_proba, tf.uint8)
        new_read = tf.cast(tf.random_uniform(shape=[read_length], maxval=4,
                                             dtype=tf.int32), tf.uint8)

        path = tf.concat([path, tf.zeros(15 - tf.size(path), dtype=tf.int32)], 0)
        path_full = tf.concat([path_full, tf.zeros(15 - tf.size(path_full), dtype=tf.int32)], 0)

        #New edit, padding the sequence
        cut_off = tf.random_uniform(shape=[1], dtype=tf.int32, maxval=9 * int(read_length / 10), minval=2)
        new_read = (1 - where_to_flip) * read + where_to_flip * new_read
        new_read = (new_read + 1)[cut_off[0]:]
        new_read = tf.concat([new_read, tf.zeros(cut_off[0], dtype=tf.uint8)], 0)

        return idx, new_read, path_full, levels
    return _parse_function

def _parse_function_tfr(hash_table, paths, paths_all, levels, read_length, old_model):
    def _parse_function(example):
        """
        Parse dataset saved in tfr record (forward, backward sequences and taxo
        label.
        """
        features = {"seq_fwd": tf.FixedLenFeature((), tf.string, default_value=""),
                    "label": tf.FixedLenFeature((), tf.string, default_value="")} 

        parsed_features = tf.parse_single_example(example, features)

        seq_fw = tf.decode_raw(parsed_features['seq_fwd'], out_type=tf.uint8)
        if old_model == 0:
            seq_fw = seq_fw + 1
            paddings = [[0, tf.reduce_max([read_length - tf.size(seq_fw), 0])]]
            seq_fw = tf.pad(seq_fw, paddings, 'CONSTANT')[0 : read_length]

        taxo = tf.decode_raw(parsed_features['label'], out_type=tf.int64)
        label = hash_table.lookup(tf.cast(taxo, tf.int32))[0]

        return label, seq_fw, paths[label], tf.constant([]), paths_all[label], levels[label], taxo

    return _parse_function


def create_dataset_from_generator(generator,
                                  batch_size,
                                  accept_proba,
                                  read_length, 
                                  buffer_size=100000,
                                  shuffle=True):
    
    """
    Given a generator function, return iterator over the resulting dataset.
    """
    dataset = tf.data.Dataset.from_generator(generator, 
                                             (tf.int32, tf.uint8, 
                                              tf.int32, tf.int32, tf.int32))
    dataset = dataset.map(_parse_function_wrap(accept_proba, read_length))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def create_dataset_from_tfr(path_to_tfr,
                            hash_table, 
                            paths,
                            paths_all,
                            levels,
                            batch_size,
                            read_length,
                            buffer_size=100,
                            shuffle=False, 
                            old_model=0):
    """
    Return dataset iterator built from tfr file. 
    """
    dataset = tf.data.TFRecordDataset(path_to_tfr)
    dataset = dataset.map(_parse_function_tfr(hash_table, paths, paths_all, levels, read_length, old_model))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat()    
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator


def build_tf_hash_table(hash_table):

    keys, values = [], []
    for k in hash_table:
        keys.append(k)
        values.append(hash_table[k])

    tf_hash_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
          keys, 
          values),
          -1)

    return tf_hash_table


def find_best_match(y_true, y_pred_list):
    """
    Find the best sequence prediction among n predictions. 

    Used for finding the best match given beam_search predictions. 
    """
    index_best = -1
    for i in range(y_pred_list.shape[-1]):
        match = y_true == y_pred_list[:, i].flatten()
        first_fail = np.where(match == False)[0]

        if len(first_fail) == 0:
            return (y_pred_list.shape[0] - 1)

        last_correct = first_fail[0] - 1
        if last_correct > index_best:
            index_best = last_correct

    return index_best


def compute_accuracy_level(y_pred, y_true_rel, y_true_abs, taxo_tree, level):
    """
    Compute accuracy at a given level of the taxonomical tree. 

    Args:
      y_pred: predictions, shape [n_samples, seq_length, n_predictions]
      y_true_rel, abs: absolute and relative paths.
      level: 'genus', 'species' or 'leaf'.
    """
    accuracy = []
    for i in range(y_pred.shape[0]):
        index_best = find_best_match(y_true_rel[i], y_pred[i])
        if level == 'leaf':
            accuracy.append(
              int(index_best == (y_pred[i].shape[0] - 1)))
            continue
        made_prediction = False
        for j, taxo in enumerate(y_true_abs[i]):
            if taxo == 0: break
            
            if taxo_tree.node_to_group[taxo] == level:
                accuracy.append(int(j <= index_best))
                made_prediction = True
                break
        # If didnt find level, add correct if full prediction is OK. 
        if not made_prediction:
            accuracy.append(int(index_best == (y_pred[i].shape[0] - 1)))

    return accuracy

def all_paths_from_labels(y_pred, taxo_tree):
    """
    Given label predictions, compute corresponding paths.
    
    Args:
      y_pred: shape [n_samples, n_predictions]
    """
    y_pred_path = []

    for i in range(y_pred.shape[0]):
        y_pred_i = []
        for num_pred in range(y_pred.shape[1]):
            y_pred_i.append(np.array(
              taxo_tree.paths[y_pred[i, num_pred]])[:,None])
        y_pred_i = np.concatenate(y_pred_i, 1)
        y_pred_path.append(y_pred_i[None, :, :])
    y_pred_path = np.concatenate(y_pred_path, 0)

    return y_pred_path + 1

def compute_accuracy_levels(y_pred, y_true):
    """
    Compute accuracy for level prediction.
    """
    ac = y_true.shape[1] * [0]

    for i in range(y_true.shape[0]):
        for group in range(len(y_pred)):
            for k in range(y_pred[group].shape[-1]):
                if y_pred[group][i, k] == y_true[i, group]:
                    ac[group] += 1

    return ac

def compute_pred_true(session, model):
    y, y_pred, y_levels, x = session.run([model.y, model.logits, model.y_levels, model.x])

    mask = np.where(y != -1)
    y_pred = [yp[mask] for yp in y_pred]
    y_levels = y_levels[mask]

    lengths = []
    for xi in x:
        c_len = np.where(xi != 0)[0][-1]
        lengths.append(c_len)

    return y_pred, y_levels, lengths

def compute_accuracy(session, model, num_mb, mode, taxo_tree):
    """
    Compute accuracy of model.
    """
    ac_all_genus, ac_all_species = 0, 0
    ac_all_leaf = 0
    predictions_made = 0
    #Make room for leaf prediction
    ac_all = (len(taxo_tree.groups) + 1) * [0]
    ac_all_5s = (len(taxo_tree.groups) + 1) * [0]

    for mb in range(num_mb):

        y_pred, y_true, y_levels = session.run(
          [model.logits, model.y, model.y_levels])

        mask = np.where(y_true != -1)

        if mode == 'sg':
            groups = taxo_tree.groups

            y_pred = [np.argsort(yp, 1)[:, ::-1] for yp in y_pred]

            ac = compute_accuracy_levels(
              [y[mask][:, 0:1] for y in y_pred], y_levels[mask])

            ac_5s = compute_accuracy_levels(
              [y[mask][:, 0:5] for y in y_pred], y_levels[mask])
           
            for i in range(len(ac)):
                ac_all[i] += ac[i]
                ac_all_5s[i] += ac_5s[i]

            predictions_made += mask[0].size

    results, results_5s = {}, {}
    for i, group in enumerate(groups):
        results[group] = ac_all[i] / float(predictions_made)
    results['leaf'] = ac_all[-1] / float(predictions_made)

    for i, group in enumerate(groups):
        results_5s[group] = ac_all_5s[i] / float(predictions_made)
    results_5s['leaf'] = ac_all_5s[-1] / float(predictions_made)

    return [results, results_5s]

def save_hist_to_tensorboard(writer, session, x, tag, mb, bins=50):
    """
    Given a list x, compute histogram, save to figure and write 
    to tensorboard (writer). 
    """
    plt.figure()
    plt.hist(x, bins=bins)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    summary = session.run(tf.summary.image(tag, image))
    writer.add_summary(summary, mb)
    plt.close()


def lr_on_plateau(validation_accuracies, lr_threshold, lr_patience, 
                  lr_decay, current_lr, min_lr = 1e-5):

    """
    Decay learning rate when on a plateau for lr_patience evaluations. 
    """

    if len(validation_accuracies) < lr_patience + 1: 
        return current_lr

    best = np.amax(validation_accuracies[-lr_patience-1:-1])
    if validation_accuracies[-1] < lr_threshold * best:
        return max(lr_decay * current_lr, min_lr)
    else:
        return current_lr


def get_conv_layer(input, strides, conv_filter_shape, name, padding='VALID'):

    conv = tf.layers.conv2d(
      inputs=input,
      filters=conv_filter_shape[-1],
      strides=strides[1:3],
      kernel_size=conv_filter_shape[0:2],
      padding=padding)
      
    return conv




