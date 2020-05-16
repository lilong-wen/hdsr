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

def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + "\\" + fn)
                    if not os.path.isdir(dir + fn) else 0)
    file = os.path.join(dir, file_lists[-1])
    return file


def test_continue():

    test_root = './test_img/'
    model_root = './src/assets/models/'
    count = 0
    for item in os.listdir(test_root):
        img = test_root + item
        image_array = cv2.imread(img)
        model_name_newerest = find_new_file(model_root)
        a = TextRecognition(model_name=model_name_newerest)
        print("using model" + model_name_newerest + "for test")
        result = a.recognize_text(image_array)

        label = item.split('-')[0]
        if result == label:
            count = count + 1
    return count 
