import keras.callbacks
import numpy as np
from src.file_loader import FileLoader
import src.text_preprocessing as text_preprocessing


class DataGenerator(keras.callbacks.Callback):
    def __init__(self, downsample_factor=2 ** 2, line_width_padded=535,
                 line_height_normalized=64, max_string_length=7):
        super().__init__()

        self.downsample_factor = downsample_factor

        self.line_height_normalized = line_height_normalized
        self.line_width_padded = line_width_padded
        self.max_string_length = max_string_length

        self.image_paths = None
        self.image_labels = None

        self.batch_size = None
        self.validation_split = None
        self.number_of_training_data = None
        self.number_of_validation_data = None
        self.number_of_validation_split = None

        self.current_train_index = None
        self.current_val_index = None

    def setup_training(self, image_paths, image_labels, batch_size=32, validation_split=0.2):
        assert len(image_paths) == len(image_labels)

        self.image_paths = image_paths
        self.image_labels = image_labels

        self.batch_size = batch_size

        self.validation_split = validation_split
        self.number_of_training_data = len(image_paths)
        self.number_of_validation_data = int(self.number_of_training_data * validation_split)
        self.number_of_validation_split = self.number_of_training_data - self.number_of_validation_data

        self.current_train_index = 0
        self.current_val_index = self.number_of_validation_split

    def generate_training(self):
        while 1:
            batch_output = self._get_batch_for_training()

            self.current_train_index += self.batch_size
            if self.current_train_index >= self.number_of_validation_split:
                self.current_train_index = self.current_train_index % 32  # probably self.batch_size
                self.image_paths, self.image_labels = self._shuffle_data(self.image_paths, self.image_labels)

            yield batch_output

    def generate_validation(self):
        while 1:
            batch_output = self._get_batch_for_training()

            self.current_val_index += self.batch_size
            if self.current_val_index >= self.number_of_training_data:
                self.current_val_index = self.number_of_validation_split + self.current_val_index % 32
                # set index back to the validation split starting value

            yield batch_output

    def generate_prediction(self, image, batch_size=1):
        while 1:
            batch_output = self._get_batch_for_prediction(image, batch_size)
            network_input = batch_output[0]
            yield batch_output

    def get_steps_per_epoch(self):
        return (self.number_of_training_data - self.number_of_validation_data) // self.batch_size

    def get_validation_steps(self):
        #print(8*"*")
        #print(self.number_of_validation_data)
        #print(self.batch_size)
        return self.number_of_validation_data // self.batch_size

    def on_train_begin(self, logs=None):
        print("On train begin...")
        self._shuffle_data(self.image_paths, self.image_labels)

    def _get_batch_for_training(self):
        batch_images, batch_labels, input_length, label_length = self._bootstrap_model_input(batch_size=self.batch_size)

        target_strings = []
        for i in range(self.batch_size):
            image_path = self.image_paths[self.current_train_index + i]
            image_label = self.image_labels[self.current_train_index + i]

            target_strings.append(image_label)

            image_train = self._get_image(image_path)
            x_train_image = text_preprocessing.process_image(image_train, self.line_height_normalized)
            x_train_image = text_preprocessing.pad_sequence(x_train_image, self.line_height_normalized,
                                                            self.line_width_padded)
            y_train_label = self._process_label(image_label)

            single_label_length = len(image_label)
            single_label_length = np.array([single_label_length])

            batch_images[i, 0:self.line_width_padded, :, 0] = x_train_image
            batch_labels[i, 0:len(image_label)] = y_train_label
            input_length[i] = self.line_width_padded // self.downsample_factor - 2
            # -2 as the first couple outputs of the RNN tend to be garbage
            label_length[i] = single_label_length

        inputs = {'input': batch_images,
                  'labels': batch_labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'target_strings': target_strings
                  }
        outputs = {'ctc': np.zeros([self.batch_size])}  # dummy data for dummy loss function
        return inputs, outputs

    def _get_batch_for_prediction(self, image, batch_size):
        batch_images, batch_labels, input_length, label_length = self._bootstrap_model_input(batch_size=batch_size)

        for i in range(batch_size):
            image_label = '42'

            x_image = text_preprocessing.process_image(image, self.line_height_normalized)
            x_image = text_preprocessing.pad_sequence(x_image, self.line_height_normalized,
                                                      self.line_width_padded)

            y_label = self._process_label(image_label)

            single_label_length = len(image_label)
            single_label_length = np.array([single_label_length])

            batch_images[i, 0:self.line_width_padded, :, 0] = x_image
            batch_labels[i, 0:len(image_label)] = y_label
            input_length[i] = self.line_width_padded // self.downsample_factor - 2
            # -2 as the first couple outputs of the RNN tend to be garbage
            label_length[i] = single_label_length

        inputs = {'input': batch_images,
                  'labels': batch_labels,
                  'input_length': input_length,
                  'label_length': label_length
                  }
        outputs = {'ctc': np.zeros([batch_size])}  # dummy data for dummy loss function
        return inputs, outputs

    def _bootstrap_model_input(self, batch_size):
        batch_images = np.ones([batch_size, self.line_width_padded, self.line_height_normalized, 1])
        batch_labels = np.ones([batch_size, self.max_string_length]) * -1
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])

        return batch_images, batch_labels, input_length, label_length

    def _process_label(self, image_label):
        ret = []
        lower_case_offset = 26
        upper_case_offset = 26
        number_offset = 10

        try:
            for char in image_label:
                if char >= 'a' and char <= 'z':
                    ret.append(ord(char) - ord('a'))  # lowercase letters go from 0 to 25
                elif char >= 'A' and char <= 'Z':
                    ret.append(ord(char) - ord('A') + lower_case_offset)
                elif char in '0123456789':
                    ret.append(ord(char) - ord('0') + lower_case_offset + upper_case_offset)
                elif char == ' ':
                    ret.append(number_offset + lower_case_offset + upper_case_offset)
                elif char == ',':
                    ret.append(number_offset + lower_case_offset + upper_case_offset + 1)
                elif char == '.':
                    ret.append(number_offset + lower_case_offset + upper_case_offset + 2)
                elif char == '-':
                    ret.append(number_offset + lower_case_offset + upper_case_offset + 3)
                elif char == '/':
                    ret.append(number_offset + lower_case_offset + upper_case_offset + 4)
                else:
                    '''All other possible characters are also recognised as '/' for now
                    we will have to change the model to fit in more characters, also,
                    other characters are all too rare in training data for now.'''
                    ret.append(number_offset + lower_case_offset + upper_case_offset + 4)
            return np.array(ret)
        except Exception:
            ret.append(number_offset + lower_case_offset + upper_case_offset + 4)
            return np.array(ret)

    def _shuffle_data(self, x_data, y_data, stop_index=None):
        data_length = len(self.image_paths)
        if stop_index is None:
            stop_index = data_length
        assert stop_index <= data_length

        a = list(range(stop_index))
        np.random.shuffle(a)
        a += list(range(stop_index, data_length))  # add unshuffled validation indices

        if isinstance(x_data, np.ndarray):
            x_data = x_data[a]
        elif isinstance(x_data, list):
            x_data = [x_data[i] for i in a]

        if isinstance(y_data, np.ndarray):
            y_data = y_data[a]
        elif isinstance(y_data, list):
            y_data = [y_data[i] for i in a]

        return x_data, y_data

    def _get_image(self, image_path):
        image_container = FileLoader.load_image(image_path)
        return image_container.get_unprocessed_image()
