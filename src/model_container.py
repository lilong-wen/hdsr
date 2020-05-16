import itertools
import numpy as np
from keras import models as KerasModels
from keras import backend as KerasBackend
from keras.layers import Lambda


class ModelContainer:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def load_model(self, path):
        self._model = KerasModels.load_model(path, custom_objects={
            'KerasBackend': KerasBackend,
            'Lamda': Lambda,
            '<lambda>': lambda truth, prediction: prediction
        })

    def predict(self, batch_images):
        prediction = self._model.predict(batch_images)
        decoded_words = self._decode_batch(prediction)
        return decoded_words

    def _decode_batch(self, batch_prediction_output):
        decoded_batch = []
        for j in range(batch_prediction_output.shape[0]):
            """ 
            Greedy Search for the most likely output token
            
            single_output = batch_prediction_output[j, 2:]  # first outputs tend to be garbage
            single_output = KerasBackend.expand_dims(single_output, axis=0)
            shape_x = int(single_output.shape[0])
            shape_y = int(single_output.shape[1])
            input_length = numpy.ones(shape_x) * shape_y  # maybe single_output.shape[1] and [0]
            decoded_output = KerasBackend.ctc_decode(single_output, input_length=input_length, greedy=True)
            best_output = KerasBackend.get_value(decoded_output[0][0])[:, :]  # maybe append [:,:]
            if best_output.ndim == 2:
                best_output = numpy.reshape(best_output, -1)
            """

            """
            Beam Search
            """
            out_best = list(np.argmax(batch_prediction_output[j, 2:],  # 2:
                                      1))  # return indexes of highest probability letters at each timestep
            out_best = [k for k, g in itertools.groupby(out_best)]  # removes adjacent duplicate indexes
            decoded_word = self._labels_to_text(out_best)
            decoded_batch.append(decoded_word)

        return decoded_batch

    def _labels_to_text(self, labels):
        ret = []

        if len(labels) == 0:
            return ""

        low_offset = 26
        upp_offset = 26
        num_offset = 10

        for c in labels:
            c = int(c)
            if c < 26:
                ret.append(chr(c + ord('a')))
            elif c < 52:
                ret.append(chr(c + ord('A') - 26))
            elif c < 62:
                ret.append(chr(c + ord('0') - 52))
            elif c == 62:
                ret.append(' ')
            elif c == 63:
                ret.append(',')
            elif c == 64:
                ret.append('.')
            elif c == 65:
                ret.append('-')
            elif c == 66:
                ret.append('/')
            elif c == 67 or c == -1:  # TODO: len(charset)
                ret.append("")
        return "".join(ret)
