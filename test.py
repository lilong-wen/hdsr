import numpy as np
import cv2
import os
# from config import get_path_to_model_dir_in_assets
from src.model_container import ModelContainer
import src.text_preprocessing as text_preprocessing


class TextRecognition():

    def __init__(self, model_name=''):
        self.model_data = {
            "line_width_padded": 535,
            "line_height_normalized": 64
        }
        self.model_container = ModelContainer()
        self._load_model(model_name)
    def _load_model(self, model_name):
        self.model_container.load_model(model_name)
    def recognize_text(self, image_array):
        image_array = text_preprocessing.process_image(image_array, self.model_data['line_height_normalized'])
        image_array = text_preprocessing.pad_sequence(image_array, self.model_data['line_height_normalized'],
                                                      self.model_data['line_width_padded'])
        batch_images = np.ones(
            [1, self.model_data['line_width_padded'], self.model_data['line_height_normalized'], 1])
        batch_images[0, 0:self.model_data['line_width_padded'], :, 0] = image_array
        if self.model_container.model is not None:
            result = self.model_container.predict(batch_images)
            #return result[0]
            return result
        else:
            print("No model was chosen")

if __name__ == '__main__':

    test_root = './test_img/'
    model_root = 'src/assets/models/'
    for item in os.listdir(test_root):
        img = test_root + item
        image_array = cv2.imread(img)
        a = TextRecognition(model_name=model_root + 'rnn_prediction_2020_05_16_21_13_40.h5')
        result = a.recognize_text(image_array)
        print(result)
        print(type(result))
        label = item.split('-')[0]
        if result != label and result != label:
            print(label + " : " + str(result))
