import os
import cv2


class ImageContainer:
    def __init__(self, path):
        self.image_path = path

        self._unprocessed_image = self._load_image(path)
        self._processed_image = None

    def _load_image(self, path):
        return cv2.imread(path)

    def get_unprocessed_image(self):
        return self._unprocessed_image

    def set_processed_image(self, image):
        self._processed_image = image

    def get_dimensions(self, from_processed_image=True):
        if from_processed_image and self._processed_image is not None:
            height, width, _ = self._processed_image.shape
            return width, height
        else:
            height, width, _ = self._unprocessed_image.shape
            return width, height

    def save_processed_image(self, output_folder):
        head, tail = os.path.split(self.image_path)
        output_path = output_folder + tail
        cv2.imwrite(output_path, self._processed_image)

        return output_path
