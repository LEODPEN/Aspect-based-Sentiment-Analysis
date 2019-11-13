# -*- coding: utf-8 -*-

import os
import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Dropout, Conv1D, MaxPool1D, Flatten, concatenate, Dense, \
    LSTM, Bidirectional, Activation, MaxPooling1D, Add, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, \
    TimeDistributed, Permute, multiply, Lambda, add, Masking, BatchNormalization, Softmax, Reshape, ReLU, \
    ZeroPadding1D, subtract, GaussianNoise, MaxoutDense
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import keras.backend as K
import tensorflow as tf

from layers import AttentionWithContext, Attention, ELMoEmbedding, MeanOverTime
from utils import get_score_senti
from data_loader import load_idx2token


# callback for sentiment analysis model
class SentiModelMetrics(Callback):
    def __init__(self):
        super(SentiModelMetrics, self).__init__()

    def on_train_begin(self, logs={}):
        self.val_accs = []
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs={}):
        if len(self.validation_data[:-3]) == 1:
            x_valid = self.validation_data[0]
        else:
            x_valid = self.validation_data[:-3]
        y_valid = self.validation_data[-3]
        valid_results = self.model.predict(x_valid)
        _val_acc, _val_f1 = get_score_senti(y_valid, valid_results)
        logs['val_acc'] = _val_acc
        logs['val_f1'] = _val_f1
        self.val_accs.append(_val_acc)
        self.val_f1s.append(_val_f1)
        print('val_acc: %f' % _val_acc)
        print('val_f1: %f' % _val_f1)
        return


# model for sentiment analysis
class SentimentModel(object):
    def __init__(self, config):
        self.config = config
        self.level = self.config.level # char
        self.use_elmo = self.config.use_elmo # false
        self.max_len = self.config.max_len[self.config.data_name][self.level] # 127
        self.asp_max_len = self.config.asp_max_len[self.config.data_name][self.level] # 19

        # 搞定embedding？
        if self.config.use_text_input:

            self.text_embeddings = np.load('./data/%s/%s_%s.npy' % (self.config.data_folder, self.level,
                                                                    self.config.word_embed_type))
            self.config.idx2token = load_idx2token(self.config.data_folder, self.level)
        else:
            self.text_embeddings = None

        if self.config.use_aspect_input:
            self.aspect_embeddings = np.load('./data/%s/aspect_%s_%s.npy' % (self.config.data_folder, self.level,
                                                                             self.config.aspect_embed_type))
            # 是否随机chushihua
            if config.aspect_embed_type == 'random':
                self.n_aspect = self.aspect_embeddings.shape[0]
                self.aspect_embeddings = None
        else:
            self.aspect_embeddings = None

        if self.config.use_aspect_text_input: # false，未用上
            self.aspect_text_embeddings = np.load('./data/%s/aspect_text_%s_%s.npy' % (self.config.data_folder,
                                                                                       self.level,
                                                                                       self.config.word_embed_type))
            self.config.idx2aspect_token = load_idx2token(self.config.data_folder, 'aspect_text_{}'.format(self.level))
        else:
            self.aspect_text_embeddings = None

        self.all_embeddings = np.load('./data/%s/all_%s_%s.npy' %  (self.config.data_folder, self.level, self.config.word_embed_type))

        self.callbacks = []
        self.init_callbacks()

        self.model = None
        self.build_model()

    def init_callbacks(self):
        self.callbacks.append(SentiModelMetrics())

        self.callbacks.append(ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, '%s/%s.hdf5' % (self.config.data_folder,
                                                                              self.config.exp_name)),
            monitor=self.config.checkpoint_monitor, # 'val_acc'
            save_best_only=self.config.checkpoint_save_best_only,
            save_weights_only=self.config.checkpoint_save_weights_only,
            mode=self.config.checkpoint_save_weights_mode, # monitor为val_acc时，模式应为max，当监测值为val_loss时，模式应为min
            verbose=self.config.checkpoint_verbose # 信息展示模式 0/1
        ))

    # 加载模型
    def load(self):
        print('loading model checkpoint {} ...\n'.format('%s.hdf5') % self.config.exp_name)
        self.model.load_weights(os.path.join(self.config.checkpoint_dir, '%s/%s.hdf5' % (self.config.data_folder,
                                                                                         self.config.exp_name)))
        print('Model loaded')

    def build_base_network(self):
        if self.config.model_name == 'atae_lstm':
            base_network = self.atae_lstm()
        elif self.config.model_name == 'tsa':
            base_network = self.tsa()
        else:
            raise Exception('Model Name `%s` Not Understood' % self.config.model_name)

        return base_network # 其实tsa时候直接给全包了

    def build_model(self):
        # 加加加
        network_inputs = list()
        # true
        if self.config.use_text_input:
            network_inputs.append(Input(shape=(self.max_len,), name='input_text'))

        # true when atae
        if self.config.use_aspect_input:
            network_inputs.append(Input(shape=(1, ), name='input_aspect'))

        # true when tsa
        if self.config.use_aspect_text_input:
            network_inputs.append(Input(shape=(self.asp_max_len,), name='input_aspect'))

        if len(network_inputs) == 1:
            network_inputs = network_inputs[0]
        elif len(network_inputs) == 0:
            raise Exception('No Input!')

        print(network_inputs)

        if self.config.model_name in ['at_lstm', 'ae_lstm', 'atae_lstm'] :
            base_network = self.build_base_network()  # atae_lstm 2
            sentence_vec = base_network(network_inputs)
            dense_layer = Dense(self.config.dense_units, activation='relu')(sentence_vec)  # 3
            output_layer = Dense(self.config.n_classes, activation='softmax')(dense_layer)  # 4

            self.model = Model(network_inputs, output_layer)
            self.model.compile(loss='categorical_crossentropy', metrics=['acc'],
                               optimizer=self.config.optimizer)  # "adam"
        elif self.config.model_name is 'tsa':
            base_network = self.build_base_network() # tsa ，直接全这里搞了算了
            output_layer = base_network(network_inputs)

            self.model = Model(network_inputs, output_layer)
            self.model.compile(loss="categorical_crossentropy", metrics=['acc'],
                          optimizer=Adam(clipnorm=self.config.clipnorm, lr=self.config.lr))
        else:
            print("到这里突然肚子好饿。。。")


    def prepare_input(self, input_data):
        if  self.config.model_name in ['at_lstm', 'ae_lstm', 'atae_lstm'] :
            text, aspect = input_data
            print(aspect)
            input_pad = [pad_sequences(text, self.max_len), np.array(aspect)]
            print(input_data) # 三维
        elif self.config.model_name is 'tsa':
            text, aspect_text = input_data
            print(aspect_text)
            input_pad = [pad_sequences(text, self.max_len),pad_sequences(aspect_text, self.asp_max_len)]
        else:
            raise ValueError('model name `{}` not understood'.format(self.config.model_name))
        return input_pad

    def prepare_label(self, label_data):
        return to_categorical(label_data, self.config.n_classes)

    def train(self, train_input_data, train_label, valid_input_data, valid_label):
        x_train = self.prepare_input(train_input_data)
        y_train = self.prepare_label(train_label)
        x_valid = self.prepare_input(valid_input_data)
        y_valid = self.prepare_label(valid_label)

        print('start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size, epochs=self.config.n_epochs,
                       validation_data=(x_valid, y_valid), callbacks=self.callbacks)
        print('training end...')

        print('score over valid data:')
        valid_pred = self.model.predict(x_valid)
        get_score_senti(y_valid, valid_pred)

    def score(self, input_data, label):
        input_pad = self.prepare_input(input_data)
        label = self.prepare_label(label)
        prediction = self.model.predict(input_pad)
        print(prediction)
        get_score_senti(label, prediction)

    def predict(self, input_data):
        input_pad = self.prepare_input(input_data)
        prediction = self.model.predict(input_pad)
        return np.argmax(prediction, axis=-1)

    # def getBiLSTM(self):
    #     # 64
    #     lstm = LSTM(self.config.lstm_size, return_sequences=True,
    #                consume_less="cpu", dropout_U=self.config.drop_text_rnn_U,
    #                W_regularizer=l2(0))
    #     return Bidirectional(lstm)


    # attention-based lstm with aspect embedding
    def atae_lstm(self):
        input_text = Input(shape=(self.max_len,))
        input_aspect = Input(shape=(1,), )

        if self.use_elmo:
            elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                           mask_zero=True, hub_url=self.config.elmo_hub_url,
                                           elmo_trainable=self.config.elmo_trainable)
            if self.config.use_elmo_alone:
                text_embed = SpatialDropout1D(0.2)(elmo_embedding(input_text))
            else:
                word_embedding = Embedding(input_dim=self.text_embeddings.shape[0],
                                           output_dim=self.config.word_embed_dim,
                                           weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                           mask_zero=True)
                text_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_text), elmo_embedding(input_text)]))
        else:
            word_embedding = Embedding(input_dim=self.text_embeddings.shape[0], output_dim=self.config.word_embed_dim,
                                       weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                       mask_zero=True)
            # dropout 丢弃比例0.2
            text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))

        if self.config.aspect_embed_type == 'random':
            asp_embedding = Embedding(input_dim=self.n_aspect, output_dim=self.config.aspect_embed_dim)
        else:
            asp_embedding = Embedding(input_dim=self.aspect_embeddings.shape[0],
                                      output_dim=self.config.aspect_embed_dim,
                                      trainable=self.config.aspect_embed_trainable)
        aspect_embed = asp_embedding(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
        repeat_aspect = RepeatVector(self.max_len)(aspect_embed)  # repeat aspect for every word in sequence

        input_concat = concatenate([text_embed, repeat_aspect], axis=-1)
        hidden_vecs, state_h, _ = LSTM(self.config.lstm_units, return_sequences=True, return_state=True)(input_concat)
        concat = concatenate([hidden_vecs, repeat_aspect], axis=-1)

        # apply attention mechanism
        attend_weight = Attention()(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        attend_hidden_dense = Dense(self.config.lstm_units)(attend_hidden)
        last_hidden_dense = Dense(self.config.lstm_units)(state_h)
        final_output = Activation('tanh')(add([attend_hidden_dense, last_hidden_dense]))

        return Model([input_text, input_aspect], final_output)

    # task 4 TSA model
    def tsa(self):
        input_text = Input(shape=(self.max_len,))
        # 以char序列加入
        input_aspect = Input(shape=(self.asp_max_len,), )

        # embedding 按照原文来--使用了一个统一的词汇表，如果分开使用text和aspect的词汇表会怎么样呢？
        text_embedding = Embedding(input_dim=self.all_embeddings.shape[0],
                                   output_dim=self.config.word_embed_dim,
                                   input_length = self.max_len,
                                   weights=[self.all_embeddings],
                                   trainable=self.config.word_embed_trainable,
                                   mask_zero=True)
        aspect_embedding = Embedding(input_dim=self.all_embeddings.shape[0],
                                     output_dim=self.config.word_embed_dim,
                                     # input_length = self.asp_max_len,
                                     weights=[self.all_embeddings],
                                     trainable=self.config.aspect_embed_trainable,
                                     mask_zero=True)

        text_embed = text_embedding(input_text)
        text_embed = GaussianNoise(self.config.noise)(text_embed)
        text_embed = Dropout(self.config.drop_text_input)(text_embed)

        aspect_embed = aspect_embedding(input_aspect)
        aspect_embed = GaussianNoise(self.config.noise)(aspect_embed)

        # bilstm
        lstm = LSTM(self.config.lstm_size, return_sequences=True,
                    consume_less="cpu", dropout_U=self.config.drop_text_rnn_U,
                    W_regularizer=l2(0))

        BiLSTM = Bidirectional(lstm)

        h_text = BiLSTM(text_embed)
        h_text = Dropout(self.config.drop_text_rnn)(h_text)

        h_aspect = BiLSTM(aspect_embed)
        h_aspect = Dropout(self.config.drop_target_rnn)(h_aspect)

        h_aspect = MeanOverTime()(h_aspect)
        # 重复text的 max_length 次以拼接
        h_aspect = RepeatVector(self.max_len)(h_aspect)

        # 拼接, 默认 -1
        representation = concatenate([h_text, h_aspect])

        # attention 层
        representation = AttentionWithContext()(representation)
        representation = Dropout(self.config.drop_att)(representation)

        representation = MaxoutDense(self.config.final_size)(representation)
        representation = Dropout(self.config.drop_final)(representation)

        probabilities = Dense(self.config.n_classes,
                              activation="softmax",
                              activity_regularizer=l2(self.config.activity_l2))(representation)

        model = Model([input_text, input_aspect], output=probabilities)
        # self.model.compile(loss="categorical_crossentropy", metrics=['acc'],
        #                    optimizer=Adam(clipnorm=self.config.clipnorm, lr=self.config.lr))

        return model

    # lstm with aspect embedding ?

    # attention-based lstm (supporting masking) ?



