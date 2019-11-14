# -*- coding: utf-8 -*-
import os
import nltk
import numpy as np
import pandas as pd
from collections import Counter
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from config import Config
from utils import pickle_dump


def load_glove_format(filename):
    word_vectors = {}
    embeddings_dim = -1
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            word_vector = np.array([float(v) for v in line[1:]]) # 256
            if len(word_vector) == 256:
                word_vectors[word] = word_vector
                # print(word_vector)
            if embeddings_dim == -1:
                embeddings_dim = len(word_vector)
    # i = 0
    # for vm in word_vectors.values():
    #     if len(vm)!=256:
    #         print(i)
    #         print(len(vm))
    #     i+=1
    assert all(len(vw) == embeddings_dim for vw in word_vectors.values())
    return word_vectors, embeddings_dim


def list_flatten(l):
    result = list()
    for item in l:
        if isinstance(item, (list, tuple)):
            result.extend(item)
        else:
            result.append(item)
    return result


def build_vocabulary(corpus, start_id=1):
    corpus = list_flatten(corpus)
    return dict((word, idx) for idx, word in enumerate(set(corpus), start=start_id))


def build_embedding(corpus, vocab, embedding_dim=300):
    # 维度还是300
    model = Word2Vec(corpus, size=embedding_dim, min_count=1, window=5, sg=1, iter=10) # 迭代十次
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab) + 2, embedding_dim), dtype='float32')

    count = 0
    for w, i in vocab.items():
        if w not in d:
            count += 1
            emb[i, :] = np.random.uniform(-0.1, 0.1, embedding_dim)
        else:
            emb[i, :] = weights[d[w], :]
    print('embedding out of vocabulary：', count)
    return emb


def build_glove_embedding(vocab, word_vectors, embed_dim): # 256
    emb_matrix = np.zeros(shape=(len(vocab) + 2, embed_dim), dtype='float32')

    count = 0
    for word, i in vocab.items():
        if word not in word_vectors:
            # 随机搞, 1804 2379 感觉这样的字还挺多的。。。
            count += 1
            emb_matrix[i, :] = np.random.uniform(-0.1, 0.1, embed_dim)
        else:
            emb_matrix[i, :] = word_vectors[word]
    print('glove embedding out of vocabulary：', count)
    return emb_matrix


def build_aspect_embedding(aspect_vocab, split_func, word_vocab, word_embed):
    aspect_embed = np.random.uniform(-0.1, 0.1, [len(aspect_vocab.keys()), word_embed.shape[1]])
    count = 0
    for aspect, aspect_id in aspect_vocab.items():
        word_ids = [word_vocab.get(word, 0) for word in split_func(aspect)]
        if any(word_ids):
            avg_vector = np.mean(word_embed[word_ids], axis=0)
            aspect_embed[aspect_id] = avg_vector
        else:
            count += 1
    print('aspect embedding out of vocabulary:', count)
    return aspect_embed


def build_aspect_text_embedding(aspect_text_vocab, word_vocab, word_embed):
    aspect_text_embed = np.zeros(shape=(len(aspect_text_vocab) + 2, word_embed.shape[1]), dtype='float32')
    count = 0
    for aspect, aspect_id in aspect_text_vocab.items():
        if aspect in word_vocab:
            aspect_text_embed[aspect_id] = word_embed[word_vocab[aspect]]
        else:
            count += 1
            aspect_text_embed[aspect_id] = np.random.uniform(-0.1, 0.1, word_embed.shape[1])
    print('aspect text embedding out of vocabulary:', count)
    return aspect_text_embed


def analyze_len_distribution(train_input, valid_input, test_input):
    text_len = list()
    text_len.extend([len(l) for l in train_input])
    text_len.extend([len(l) for l in valid_input])
    text_len.extend([len(l) for l in test_input])
    max_len = np.max(text_len)
    min_len = np.min(text_len)
    avg_len = np.average(text_len)
    median_len = np.median(text_len)
    print('max len:', max_len, 'min_len', min_len, 'avg len', avg_len, 'median len', median_len)
    for i in range(int(median_len), int(max_len), 5):
        less = list(filter(lambda x: x <= i, text_len))
        ratio = len(less) / len(text_len)
        print(i, ratio)
        if ratio >= 0.99:
            break


def analyze_class_distribution(labels):
    for cls, count in Counter(labels).most_common():
        print(cls, count, count / len(labels))



def pre_process_Car(file_folder, word_cut_func = None):
    print('preprocessing: ', file_folder)
    # 读取数据
    train_data = pd.read_csv(os.path.join(file_folder, 'train_text_cate.tsv'), sep=config.sep, header=None, index_col=None)
    train_data['char_list'] = train_data[0].apply(lambda x: list(x))
    train_data['aspect_char_list'] = train_data[1].apply(lambda x: list(x))

    valid_data = pd.read_csv(os.path.join(file_folder, 'dev_text_cate.tsv'), sep=config.sep, header=None, index_col=None)
    valid_data['char_list'] = valid_data[0].apply(lambda x: list(x))
    valid_data['aspect_char_list'] = valid_data[1].apply(lambda x: list(x))

    test_data = pd.read_csv(os.path.join(file_folder, 'test_text_cate.tsv'), sep=config.sep, header=None, index_col=None)
    test_data['char_list'] = test_data[0].apply(lambda x: list(x))
    test_data['aspect_char_list'] = test_data[1].apply(lambda x: list(x))

    print('size of training set:', len(train_data))
    print('size of valid/dev set:', len(valid_data))
    print('size of test set:', len(test_data))

    # char 集
    char_corpus = np.concatenate((train_data['char_list'].values, valid_data['char_list'].values,
                                  test_data['char_list'].values)).tolist()

    # aspect 集
    aspect_corpus = np.concatenate((train_data[1].values, valid_data[1].values,
                                    test_data[1].values)).tolist()

    # aspect char 集
    aspect_text_char_corpus = np.concatenate((train_data['aspect_char_list'].values,
                                              valid_data['aspect_char_list'].values,
                                              test_data['aspect_char_list'].values)).tolist()

    # all char 集, 主要用于tsa model
    all_char_corpus = np.concatenate((train_data['char_list'].values,
                                      valid_data['char_list'].values,
                                      test_data['char_list'].values,
                                      train_data['aspect_char_list'].values,
                                      valid_data['aspect_char_list'].values,
                                      test_data['aspect_char_list'].values)).tolist()


    # build vocabulary
    print('building vocabulary...')
    # 变成字典
    char_vocab = build_vocabulary(char_corpus, start_id=1)
    aspect_vocab = build_vocabulary(aspect_corpus, start_id=0)  # mask_zero = false

    aspect_text_char_vocab = build_vocabulary(aspect_text_char_corpus, start_id=1)

    all_char_vocab = build_vocabulary(all_char_corpus, start_id=1)

    pickle_dump(char_vocab, os.path.join(file_folder, 'char_vocab.pkl'))
    pickle_dump(aspect_vocab, os.path.join(file_folder, 'aspect_vocab.pkl'))
    pickle_dump(aspect_text_char_vocab, os.path.join(file_folder, 'aspect_text_char_vocab.pkl'))
    pickle_dump(all_char_vocab, os.path.join(file_folder, 'all_char_vocab.pkl'))

    print('finished building 4 vocabulary sets!')
    print('len of char vocabulary:', len(char_vocab))
    print('sample of char vocabulary:', list(char_vocab.items())[:10])
    print('len of aspect vocabulary:', len(aspect_vocab))
    print('sample of aspect vocabulary:', list(aspect_vocab.items())[:10])
    print('len of aspect text char vocabulary:', len(aspect_text_char_vocab))
    print('sample of aspect text char vocabulary:', list(aspect_text_char_vocab.items())[:10])
    print('len of all char vocabulary:', len(all_char_vocab))
    print('sample of all char vocabulary:', list(all_char_vocab.items())[:10])


    # prepare embedding
    print('preparing embedding...')
    char_w2v = build_embedding(char_corpus, char_vocab, config.word_embed_dim) # 300
    aspect_char_w2v = build_aspect_embedding(aspect_vocab, lambda x: list(x), char_vocab, char_w2v)
    aspect_text_char_w2v = build_aspect_text_embedding(aspect_text_char_vocab, char_vocab, char_w2v)
    all_char_w2v = build_embedding(all_char_corpus, all_char_vocab, config.word_embed_dim)
    np.save(os.path.join(file_folder, 'char_w2v.npy'), char_w2v)
    np.save(os.path.join(file_folder, 'aspect_char_w2v.npy'), aspect_char_w2v)
    np.save(os.path.join(file_folder, 'aspect_text_char_w2v.npy'), aspect_text_char_w2v)
    np.save(os.path.join(file_folder, 'all_char_w2v.npy'), all_char_w2v)

    print('finished preparing embedding!')
    print('shape of char_w2v:', char_w2v.shape)
    print('sample of char_w2v:', char_w2v[:2, :5])

    print('shape of aspect_char_w2v:', aspect_char_w2v.shape)
    print('sample of aspect_char_w2v:', aspect_char_w2v[:2, :5])

    print('shape of aspect_text_char_w2v:', aspect_text_char_w2v.shape)
    print('sample of aspect_text_char_w2v:', aspect_text_char_w2v[:2, :5])

    print('shape of all_char_w2v:', all_char_w2v.shape)
    print('sample of all_char_w2v:', all_char_w2v[:2, :5])

    # use glove
    if config.word_embed_type == 'glove':
        char_glove = build_glove_embedding(char_vocab, glove_vectors, glove_embed_dim)
        all_char_glove = build_glove_embedding(all_char_vocab, glove_vectors, glove_embed_dim)
        np.save(os.path.join(file_folder, 'char_glove.npy'), char_glove)
        np.save(os.path.join(file_folder, 'all_char_glove.npy'), all_char_glove)
        print('shape of char_glove:', char_glove.shape)
        print('sample of char_glove:', char_glove[:2, :5])
        print('shape of all_char_glove:', all_char_glove.shape)
        print('sample of all-char_glove:', all_char_glove[:2, :5])

    # prepare input
    print('preparing text input...')
    train_char_input = train_data['char_list'].apply(
        lambda x: [char_vocab.get(char, len(char_vocab)+1) for char in x]).values.tolist()
    valid_char_input = valid_data['char_list'].apply(
        lambda x: [char_vocab.get(char, len(char_vocab) + 1) for char in x]).values.tolist()
    test_char_input = test_data['char_list'].apply(
        lambda x: [char_vocab.get(char, len(char_vocab) + 1) for char in x]).values.tolist()

    pickle_dump(train_char_input, os.path.join(file_folder, 'train_char_input.pkl'))
    pickle_dump(valid_char_input, os.path.join(file_folder, 'valid_char_input.pkl'))
    pickle_dump(test_char_input, os.path.join(file_folder, 'test_char_input.pkl'))

    print('finished preparing text input!')
    print('length analysis of text char input')
    analyze_len_distribution(train_char_input, valid_char_input, test_char_input)

    print('preparing aspect input...')
    train_aspect_input = train_data[1].apply(lambda x: [aspect_vocab[x]]).values.tolist()
    valid_aspect_input = valid_data[1].apply(lambda x: [aspect_vocab[x]]).values.tolist()
    test_aspect_input = test_data[1].apply(lambda x: [aspect_vocab[x]]).values.tolist()
    pickle_dump(train_aspect_input, os.path.join(file_folder, 'train_aspect_input.pkl'))
    pickle_dump(valid_aspect_input, os.path.join(file_folder, 'valid_aspect_input.pkl'))
    pickle_dump(test_aspect_input, os.path.join(file_folder, 'test_aspect_input.pkl'))
    print('finished preparing aspect input!')

    print('preparing aspect text input...')
    train_aspect_text_char_input = train_data['aspect_char_list'].apply(
        lambda x: [aspect_text_char_vocab.get(char, len(aspect_text_char_vocab) + 1) for char in x]).values.tolist()
    valid_aspect_text_char_input = valid_data['aspect_char_list'].apply(
        lambda x: [aspect_text_char_vocab.get(char, len(aspect_text_char_vocab) + 1) for char in x]).values.tolist()
    test_aspect_text_char_input = test_data['aspect_char_list'].apply(
        lambda x: [aspect_text_char_vocab.get(char, len(aspect_text_char_vocab) + 1) for char in x]).values.tolist()
    pickle_dump(train_aspect_text_char_input, os.path.join(file_folder, 'train_char_aspect_input.pkl'))
    pickle_dump(valid_aspect_text_char_input, os.path.join(file_folder, 'valid_char_aspect_input.pkl'))
    pickle_dump(test_aspect_text_char_input, os.path.join(file_folder, 'test_char_aspect_input.pkl'))

    print('finished preparing aspect text input!')
    print('length analysis of aspect text char input') # 占比
    analyze_len_distribution(train_aspect_text_char_input, valid_aspect_text_char_input, test_aspect_text_char_input)

    # prepare output
    print('preparing output....')
    train_label_data = pd.read_csv(os.path.join(file_folder, 'train_senti.tsv'), sep=config.sep, header=None, index_col=None)
    valid_label_data = pd.read_csv(os.path.join(file_folder, 'dev_senti.tsv'), sep=config.sep, header=None, index_col=None)
    test_label_data = pd.read_csv(os.path.join(file_folder, 'test_senti.tsv'), sep=config.sep, header=None, index_col=None)
    pickle_dump(train_label_data[0].values.tolist(), os.path.join(file_folder, 'train_label.pkl'))
    pickle_dump(valid_label_data[0].values.tolist(), os.path.join(file_folder, 'valid_label.pkl'))
    pickle_dump(test_label_data[0].values.tolist(), os.path.join(file_folder, 'test_label.pkl'))
    print('finished preparing output!')

    print('class analysis of training set:')
    analyze_class_distribution(train_label_data[0].values.tolist())
    print('class analysis of valid set:')
    analyze_class_distribution(valid_label_data[0].values.tolist())
    print('class analysis of test set:')
    analyze_class_distribution(test_label_data[0].values.tolist())


if __name__ == '__main__':
    config = Config()
    glove_vectors, glove_embed_dim = load_glove_format('./data/vectors.txt')  # 1084 * 256
    print(glove_embed_dim)

    pre_process_Car('./data/car')
