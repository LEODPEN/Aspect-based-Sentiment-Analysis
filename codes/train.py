# -*- coding: utf-8 -*-

import os
import time
from config import Config
from data_loader import load_input_data, load_label
from models import SentimentModel

# choose
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # MacOs


def train_model(data_folder, data_name, level, model_name, is_aspect_term=True):
    config.data_folder = data_folder
    config.data_name = data_name
    # 新建存处
    if not os.path.exists(os.path.join(config.checkpoint_dir, data_folder)):
        os.makedirs(os.path.join(config.checkpoint_dir, data_folder))
    config.level = level # char 中文
    config.model_name = model_name # atae_lstm or tsa
    config.is_aspect_term = is_aspect_term # true
    config.init_input()
    # 给保存时候的名字
    config.exp_name = '{}_{}_wv_{}'.format(model_name, level, config.word_embed_type)
    # 可更新
    config.exp_name = config.exp_name + '_update' if config.word_embed_trainable else config.exp_name + '_fix'
    if config.use_aspect_input:
        config.exp_name += '_aspv_{}'.format(config.aspect_embed_type)
        config.exp_name = config.exp_name + '_update' if config.aspect_embed_trainable else config.exp_name + '_fix'
    # 不用 ，否则tensorflow_hub问题难解决？
    # if config.use_elmo:
    #     config.exp_name += '_elmo_alone_{}_mode_{}_{}'.format(config.use_elmo_alone, config.elmo_output_mode,
    #                                                           'update' if config.elmo_trainable else 'fix')

    print(config.exp_name)

    # 建
    model = SentimentModel(config)

    test_input = load_input_data(data_folder, 'test', level, config.use_text_input,
                                 config.use_aspect_input,config.use_aspect_text_input)
    test_label = load_label(data_folder, 'test')

    print(test_input)

    dev_input = load_input_data(data_folder, 'valid', level, config.use_text_input,
                                 config.use_aspect_input,config.use_aspect_text_input)

    dev_label = load_label(data_folder, 'valid')

    print(dev_input)

    # 无现有模型，开始训练
    if not os.path.exists(os.path.join(config.checkpoint_dir, '%s/%s.hdf5' % (data_folder, config.exp_name))):
        start_time = time.time()

        train_input = load_input_data(data_folder, 'train', level, config.use_text_input,
                                      config.use_aspect_input, config.use_aspect_text_input)

        train_label = load_label(data_folder, 'train')
        valid_input = load_input_data(data_folder, 'valid', level, config.use_text_input,
                                      config.use_aspect_input, config.use_aspect_text_input)
        valid_label = load_label(data_folder, 'valid')

        # train
        model.train(train_input, train_label, valid_input, valid_label)

        elapsed_time = time.time() - start_time
        print('training time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # load the best model
    model.load()

    print('score over test data...')
    model.score(dev_input, dev_label)
    model.score(test_input, test_label)




if __name__ == '__main__':
    config = Config()

    config.use_elmo = False
    config.use_elmo_alone = False
    config.elmo_trainable = False

    config.word_embed_trainable = True
    config.aspect_embed_trainable = True

    # train_model('car', 'car', 'char', 'atae_lstm')
    train_model('car', 'car', 'char', 'tsa')


    # others
    # config.word_embed_trainable = False
    # config.aspect_embed_trainable = True
    #
    # train_model('car', 'car', 'char', 'atae_lstm')
    # train_model('car', 'car', 'char', 'tsa')
    #
    #
    # config.word_embed_trainable = False
    # config.aspect_embed_trainable = False
    #
    # train_model('car', 'car', 'char', 'atae_lstm')
    # train_model('car', 'car', 'char', 'tsa')
