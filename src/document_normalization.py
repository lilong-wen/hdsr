import cv2


class DocumentNormalization:
    def __init__(self):
        pass

    @staticmethod
    def scale_x(image, width):
        width = int(width)
        scale_factor = (width / float(image.shape[1]))
        height = int(scale_factor * image.shape[0])
        return cv2.resize(image, (width, height))

    @staticmethod
    def scale_y(image, height):
        height = int(height)
        scale_factor = (height / float(image.shape[0]))
        width = int(scale_factor * image.shape[1])
        return cv2.resize(image, (width, height))

    @staticmethod
    def scale_to_value(image, value):
        height = image.shape[0]
        width = image.shape[1]

        scale_factor = None
        if height > width:
            scale_factor = value / height
        else:
            scale_factor = value / width

        return cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
