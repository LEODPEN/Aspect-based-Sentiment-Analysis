# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer

# modified based on `https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2`
class Attention(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
 e: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None, W_constraint=None,
                 u_constraint=None, b_constraint=None, use_W=True, use_bias=False, return_self_attend=False,
                 return_attend_weight=True, **kwargs):
        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.use_W = use_W
        self.use_bias = use_bias
        self.return_self_attend = return_self_attend    # whether perform self attention and return it
        self.return_attend_weight = return_attend_weight    # whether return attention weight
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        if self.use_W:
            self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),  initializer=self.init,
                                     name='{}_W'.format(self.name), regularizer=self.W_regularizer,
                                     constraint=self.W_constraint)
        if self.use_bias:
            self.b = self.add_weight(shape=(input_shape[1],), initializer='zero', name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer, constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],), initializer=self.init, name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer, constraint=self.u_constraint)
        
        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if self.use_W:
            x = K.tanh(K.dot(x, self.W))

        ait = Attention.dot_product(x, self.u)
        if self.use_bias:
            ait += self.b

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        if self.return_self_attend:
            attend_output = K.sum(x * K.expand_dims(a), axis=1)
            if self.return_attend_weight:
                return [attend_output, a]
            else:
                return attend_output
        else:
            return a

    def compute_output_shape(self, input_shape):
        if self.return_self_attend:
            if self.return_attend_weight:
                return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
            else:
                return input_shape[0], input_shape[-1]
        else:
            return input_shape[0], input_shape[1]

    @staticmethod
    def dot_product(x, kernel):
        """
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow
        Args:
            x (): input
            kernel (): weights
        Returns:
        """
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)

class AttentionWithContext(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False, **kwargs):

        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = AttentionWithContext.dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        # ait = K.dot(uit, self.u)
        ait = AttentionWithContext.dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]

    @staticmethod
    def dot_product(x, kernel):
        """
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow
        Args:
            x (): input
            kernel (): weights
        Returns:
        """
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)

class MeanOverTime(Layer):
    """
    Layer that computes the mean of timesteps returned from an RNN and supports masking
    Example:
        activations = LSTM(64, return_sequences=True)(words)
        mean = MeanOverTime()(activations)
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MeanOverTime, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, 'float32')
            return K.cast(K.sum(x, axis=1) / K.sum(mask, axis=1, keepdims=True),
                          K.floatx())
        else:
            return K.mean(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_mask(self, input, input_mask=None):
        return None

class ELMoEmbedding(Layer):
    """
    integrate ELMo Embeddings from tensorflow hub into a custom Keras layer, supporting weight update
    reference:  https://github.com/strongio/keras-elmo
                https://github.com/JHart96/keras_elmo_embedding_layer/blob/master/elmo.py
                https://tfhub.dev/google/elmo/2
    """
    def __init__(self, output_mode, idx2word=None, max_length=None, mask_zero=False, hub_url=None, elmo_trainable=None,
                 **kwargs):
        """
        inputs to ELMoEmbedding can be untokenzied sentences (shaped [batch_size, 1], typed string) or tokenzied word's
        id sequences (shaped [batch_size, max_length], typed int).
        When use untokenized sentences as input, max_length must be provided.
        When use word id sequences as input, idx2word must be provided to convert word id to word.
        """
        self.output_mode = output_mode
        if self.output_mode not in ['word_embed', 'lstm_outputs1', 'lstm_outputs2', 'elmo', 'default']:
            raise ValueError('Output Type Not Understood:`{}`'.format(self.output_mode))
        self.idx2word = idx2word
        self.max_length = max_length
        self.mask_zero = mask_zero
        self.dimension = 1024

        self.input_type = None
        self.word_mapping = None
        self.lookup_table = None

        # load elmo model locally by providing a local path due to the huge delay of downloading the model
        # for more information, see:
        # https://stackoverflow.com/questions/50322001/how-to-save-load-a-tensorflow-hub-module-to-from-a-custom-path
        # https://www.tensorflow.org/hub/hosting
        if hub_url is not None:
            self.hub_url = hub_url
        else:
            self.hub_url = 'https://tfhub.dev/google/elmo/2'
        if elmo_trainable is not None:
            self.elmo_trainable = elmo_trainable
        else:
            self.elmo_trainable = True if self.output_mode == 'elmo' else False

        self.elmo = None

        super(ELMoEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape[1] == 1:
            self.input_type = 'sentence'
            assert self.max_length is not None
        else:
            self.input_type = 'word_id'
            if self.max_length is None:
                self.max_length = input_shape[1]
            assert self.idx2word is not None
            self.idx2word[0] = ''   # padded position, must add
            self.word_mapping = [x[1] for x in sorted(self.idx2word.items(), key=lambda x: x[0])]
            self.lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(self.word_mapping,
                                                                                    default_value="<UNK>")
            self.lookup_table.init.run(session=K.get_session())

        print('Logging Info - Loading elmo from tensorflow hub....')
        self.elmo = hub.Module(self.hub_url, trainable=self.elmo_trainable, name="{}_elmo_hub".format(self.name))

        if self.elmo_trainable:
            print('Logging Info - ELMo model trainable')
            self.trainable_weights += K.tf.trainable_variables(scope="^{}_elmo_hub/.*".format(self.name))
        else:
            print('Logging Info - ELMo model untrainable')

    def call(self, inputs, mask=None):
        if self.input_type == 'sentence':
            # inputs are untokenized sentences
            embeddings = self.elmo(inputs=K.squeeze(K.cast(inputs, tf.string), axis=1),
                                   signature="default", as_dict=True)[self.output_mode]
            elmo_max_length = K.int_shape(embeddings)[1]
            if self.max_length > elmo_max_length:
                embeddings = K.temporal_padding(embeddings, padding=(0, self.max_length-elmo_max_length))
            elif elmo_max_length > self.max_length:
                # embeddings = tf.slice(embeddings, begin=[0, 0, 0], size=[-1, self.max_length, -1])
                embeddings = embeddings[:, :self.max_length, :]     # more pythonic
        else:
            # inputs are tokenized word id sequence
            # convert inputs to word sequence
            inputs = tf.cast(inputs, dtype=tf.int64)
            sequence_lengths = tf.cast(tf.count_nonzero(inputs, axis=1), dtype=tf.int32)
            embeddings = self.elmo(inputs={'tokens': self.lookup_table.lookup(inputs),
                                           'sequence_len': sequence_lengths},
                                   signature="tokens", as_dict=True)[self.output_mode]
            if self.output_mode != 'defalut':
                output_mask = K.expand_dims(K.cast(K.not_equal(inputs, 0), tf.float32), axis=-1)
                embeddings *= output_mask

        return embeddings

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero or self.input_type == 'sentence' or self.output_mode == 'default':
            # hard to compute mask when using sentences as input
            return None
        output_mask = K.not_equal(inputs, 0)
        return output_mask

    def compute_output_shape(self, input_shape):
        if self.output_mode == 'default':
            return input_shape[0], self.dimension
        else:
            return input_shape[0], self.max_length, self.dimension




