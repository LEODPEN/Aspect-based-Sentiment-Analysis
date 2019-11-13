# -*- coding: utf-8 -*-

import os
from utils import pickle_load


def load_input_data(data_folder, data_kind, level, use_text_input,
                    use_aspect_input, use_aspect_text_input):
    dirname = os.path.join('./data', data_folder) # ./data/car/
    input_data = []
    if use_text_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_input.pkl'.format(data_kind, level))))
    if use_aspect_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_aspect_input.pkl'.format(data_kind))))
    if use_aspect_text_input:
        input_data.append(pickle_load(os.path.join(dirname, '{}_{}_aspect_input.pkl'.format(data_kind, level))))

    if len(input_data) == 1:
        input_data = input_data[0]
    if len(input_data) == 0:
        raise Exception('No Input!')
    return input_data


def load_label(data_folder, data_kind):
    dirname = os.path.join('./data', data_folder)
    return pickle_load(os.path.join(dirname, '{}_label.pkl'.format(data_kind)))


def load_vocab(data_folder, vocab_type):
    dirname = os.path.join('./data', data_folder)
    return pickle_load(os.path.join(dirname, vocab_type+'_vocab.pkl'))


# 可惜了没用上
def load_idx2token(data_folder, vocab_type):
    vocab = load_vocab(data_folder, vocab_type)
    return dict((idx, word) for word, idx in vocab.items())

