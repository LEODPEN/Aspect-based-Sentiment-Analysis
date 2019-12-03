# -*- coding: utf-8 -*-

class Config(object):
    def __init__(self):

        # process
        self.sep = '\t'

        # input configuration
        self.data_folder = 'car' #'twitter'
        self.data_name = 'car' #'twitter'
        self.level = 'char'     # options are 'word' & 'char'
        self.max_len = {'car': {'word': 111, 'char': 127}, # word 就没做最大长度嗷
                        'twitter': {'word': 73, 'char': 188}}
        self.asp_max_len = {'car': {'word': 3, 'char': 19}, # word 同样没做最大长度, 其实char做了好想也没有用; 11.12 更新：tsa要用到，果然有用！
                            'twitter': {'word': 3, 'char': 21}}
        self.word_embed_dim = 256  #300
        self.text_random_input_dim = 2380  # 2378 + 2
        # self.aspect_char_random_input_dim = 71 # 69 + 2 tsa的random时候用
        self.aspect_random_input_dim = 20  # atae_lstm random用
        self.all_random_input_dim = 2381  # 2379 + 2
        self.word_embed_trainable = False
        self.word_embed_type =  'random'#'glove' #'random' #'w2v'
        self.aspect_embed_dim = 256  # 300
        self.aspect_embed_trainable = False
        self.aspect_embed_type = 'random' #'glove' #'random' #'w2v'
        self.use_text_input = False
        self.use_aspect_input = False
        self.use_aspect_text_input = False

        # atae-lstm

        # model structure configuration
        self.exp_name = None
        self.model_name = None
        self.lstm_units = 300
        self.dense_units = 128

        # model training configuration
        self.batch_size = 32
        self.n_epochs = 25 #8 #25 #50
        self.n_classes = 3

        self.dropout = 0.2
        self.learning_rate = 0.001  # lr倒是都一样
        self.optimizer = "adam"

        # tsa

        # Gaussian noise with σ = 0.2 at the em- bedding layer of both inputs // 不用后面句子 dropout of 0.3 at the embedding layer of the message,
        # dropout of 0.2 at the LSTM layer and the recurrent connec- tion of the LSTM layer and dropout of 0.3 at the attention layer and the Maxout layer.
        # Finally, we add L2 regularization of 0.001 at the loss function.
        self.noise = 0.2
        self.activity_l2 = 0.001
        self.drop_text_rnn_U = 0.2
        self.drop_text_input = 0.3
        self.drop_text_rnn = 0.2 # 0.3
        self.drop_target_rnn = 0.2
        self.final_size = 64
        self.drop_att = 0.3
        self.drop_final = 0.5
        self.lstm_size = 64 # 64
        self.lr = 0.001
        self.rnn_cells = 64
        self.clipnorm = .1

        # model saving configuration
        self.checkpoint_dir = './ckpt'
        self.checkpoint_monitor = 'val_acc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1 # 显示

        # early stopping configuration ， not use here
        self.early_stopping_monitor = 'val_acc'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1
        self.early_stopping_mode = 'max'

        # elmo embedding configure , not use here
        self.use_elmo = False
        self.use_elmo_alone = False
        self.elmo_hub_url = './raw_data/tfhub_elmo_2'
        self.elmo_output_mode = 'elmo'
        self.idx2token = None
        self.idx2aspect_token = None
        self.elmo_trainable = False

    def init_input(self):
        if  self.model_name in ['at_lstm', 'ae_lstm', 'atae_lstm']:
            self.use_text_input = True
            self.use_aspect_input = True
        elif self.model_name is 'tsa':
            self.use_text_input = True
            self.use_aspect_text_input = True
        else:
            raise ValueError('model name `{}` not exists'.format(self.model_name))

        print(self.use_text_input)
        print(self.use_aspect_input)
        print(self.use_aspect_text_input)




