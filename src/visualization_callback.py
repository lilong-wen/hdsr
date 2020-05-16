import itertools
import numpy
import editdistance
import keras.callbacks


class VisualizationCallback(keras.callbacks.Callback):
    def __init__(self, test_function, generate_validation):
        self.test_function = test_function
        self.generate_validation = generate_validation

    def on_epoch_end(self, epoch, logs=None):
        self._compute_edit_distance(64)

    def _compute_edit_distance(self, number_of_samples_to_display):
        mean_norm_edit_distance = 0.0
        mean_edit_distance = 0.0
        number_left = number_of_samples_to_display

        while number_left > 0:
            batch_of_words = next(self.generate_validation)[0]  # get only the inputs
            number_to_process = min(batch_of_words['input'].shape[0], number_left)  # size of batch or remaining words
            image_samples = batch_of_words['input'][0:number_to_process]

            batch_prediction_output = self.test_function([image_samples])[0]  # processed images as input
            decoded_words = self._decode_batch(batch_prediction_output)
            for i in range(number_to_process):
                print("Target word is {} and recognized {}".format(batch_of_words['target_strings'][i],
                                                                   decoded_words[i]))
                edit_dist = editdistance.eval(decoded_words[i], batch_of_words['target_strings'][i])
                mean_edit_distance += float(edit_dist)
                mean_norm_edit_distance += float(edit_dist) / len(batch_of_words['target_strings'][i])
            number_left -= number_to_process
        mean_norm_edit_distance = mean_norm_edit_distance / number_of_samples_to_display
        mean_edit_distance = mean_edit_distance / number_of_samples_to_display
        print('\nOut of {:d} samples:  Mean edit distance: {:.3f} Mean normalized edit distance: {:0.3f}'.format(
            number_of_samples_to_display, mean_edit_distance, mean_norm_edit_distance))

    def _decode_batch(self, batch_prediction_output):
        decoded_batch = []
        for j in range(batch_prediction_output.shape[0]):
            """ greedy search for the most likely output token
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

            # beam search
            out_best = list(numpy.argmax(batch_prediction_output[j, 2:],  # 2:
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
