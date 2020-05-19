import datetime
import os

from keras import backend as KerasBackend
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers import Activation, Dense, GRU, Input, Lambda, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from src.classifier_model import Classifier
from src.config import get_path_to_model_dir_in_assets


class RecurrentNeuralNetwork(Classifier):
    def __init__(self, charset, max_string_length=120, line_width_padded=993, line_height_normalized=33,
                 number_of_memory_units=512, pool_size=2):
        super(RecurrentNeuralNetwork, self).__init__(charset)

        self.model_name = "rnn"
        self.number_of_memory_units = number_of_memory_units
        self.pool_size = pool_size

        self.x_train = self.y_train = None
        self.x_test = self.y_test = None

        self.test_function = None
        self.line_height_normalized = line_height_normalized  # line height in pixel
        self.line_width_padded = line_width_padded
        self.max_string_length = max_string_length  # numbers of characters per line

        self._create_reccurent_nn()

    def _create_reccurent_nn(self):
        dropout = 0  # TODO: Dropout
        # batch_size is not necessary in Keras
        input_shape = (self.line_width_padded, self.line_height_normalized, 1)

        activation_function = 'relu'
        convolution_filters = 16
        kernel_size = (3, 3)
        time_dense_size = 32

        network_input = Input(name='input', shape=input_shape, dtype='float32')
        inner = Conv2D(convolution_filters, kernel_size, padding='same',
                       activation=activation_function, kernel_initializer='he_normal',
                       name='conv1')(network_input)
        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='max1')(inner)
        inner = Conv2D(convolution_filters, kernel_size, padding='same',
                       activation=activation_function, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='max2')(inner)

        convolution_to_rnn_dimensions = (
            self.line_width_padded // (self.pool_size ** 2),
            (self.line_height_normalized // (self.pool_size ** 2)) * convolution_filters)
        inner = Reshape(target_shape=convolution_to_rnn_dimensions, name='reshape')(inner)

        # cuts down input size going into rnn
        inner = Dense(time_dense_size, activation=activation_function, name='dense1')(inner)

        gru_layer_1 = GRU(self.number_of_memory_units, return_sequences=True, kernel_initializer='he_normal',
                          name='gru_layer_1', dropout=dropout, recurrent_dropout=dropout)(
            inner)
        gru_layer_1_backwards = GRU(self.number_of_memory_units, return_sequences=True, go_backwards=True,
                                    kernel_initializer='he_normal',
                                    name='gru_layer_1_backwards', dropout=dropout, recurrent_dropout=dropout)(
            inner)
        gru_layer_1_merged = add([gru_layer_1, gru_layer_1_backwards])
        gru_layer_2 = GRU(self.number_of_memory_units, return_sequences=True, kernel_initializer='he_normal',
                          name='gru_layer_2')(
            gru_layer_1_merged)
        gru_layer_2_backwards = GRU(self.number_of_memory_units, return_sequences=True, go_backwards=True,
                                    kernel_initializer='he_normal',
                                    name='gru_layer_2_backwards')(gru_layer_1_merged)
        output_layer = Dense(self.charset + 1, kernel_initializer='he_normal',
                             name='dense_layer')(concatenate([gru_layer_2, gru_layer_2_backwards]))
        #  output of the dense_layer should be the max charset plus 1 (the blank character)
        prediction = Activation('softmax', name='output_to_ctc')(output_layer)
        #  final rnn model
        self.model_for_prediction = Model(inputs=network_input, outputs=prediction)
        self.model_for_prediction.summary()

        # create the ctc layer to decode the output of the rnn
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        labels = Input(name='labels', shape=[self.max_string_length], dtype='int64')
        loss_out = Lambda(RecurrentNeuralNetwork._ctc_function, output_shape=(1,), name='ctc')(
            [prediction, labels, input_length, label_length])
        model = Model(inputs=[network_input, labels, input_length, label_length], outputs=loss_out)

        # training the ctc network with gradient descent
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        model.compile(loss={'ctc': lambda truth, prediction: prediction}, optimizer=sgd)

        self.test_function = KerasBackend.function([network_input], [prediction])

        self.model = model

    def train(self, generator, epochs, callbacks=None, start_epoch=0):
        if callbacks is None:
            callbacks = []
        history = self.model.fit_generator(generator=generator.generate_training(),
                                           validation_data=generator.generate_validation(),
                                           steps_per_epoch=generator.get_steps_per_epoch(), epochs=epochs,
                                           initial_epoch=start_epoch, validation_steps=generator.get_validation_steps(),
                                           callbacks=callbacks)
        self._save_model()
        return history

    def get_test_function(self):
        return self.test_function

    def get_model(self):
        return self.model

    def _save_model(self):
        if self.should_model_be_saved is True:
            time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            assets_dir = get_path_to_model_dir_in_assets()
            filename = self.model_name + '_prediction_' + time + ".h5"

            # Not needed : self.model.save(self.model_name + '_' + time + ".h5")
            self.model_for_prediction.save(os.path.join(assets_dir, filename))

    @staticmethod
    def _loss_dummy(truth, prediction):
        return prediction

    @staticmethod
    def _ctc_function(args):
        y_prediction, labels, input_length, label_length = args
        # y_prediction: tensor (samples, time_steps, num_categories) containing the prediction (output of the softmax).
        # y_truth: tensor (samples, max_string_length) containing the truth y_truth.
        # input_length: tensor (samples, 1) containing the sequence length for each batch item in y_prediction.
        # label_length: tensor (samples, 1) containing the sequence length for each batch item in y_truth.
        # the 2 is critical here since the first couple outputs of the rnn tend to be garbage:
        y_prediction = y_prediction[:, 2:, :]
        return KerasBackend.ctc_batch_cost(labels, y_prediction, input_length, label_length)
