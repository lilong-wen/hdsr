import cv2


class DocumentBinarization:
    def __init__(self):
        pass

    @staticmethod
    def binarize(image):
        if image is None:
            return None
        else:
            image_edited = DocumentBinarization.convert_to_grayscale(image)
            image_edited = DocumentBinarization.threshold_image(image_edited)
            return image_edited

    @staticmethod
    def convert_to_grayscale(image):
        image_grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image_grayscaled

    @staticmethod
    def threshold_image(image):
        block_size = 19  # Size of a pixel neighborhood that is used to calculate a threshold value
        constant_subtracted_from_mean = 4  # Normally, it is positive but may be zero or negative as well.
        image_thresholed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, block_size, constant_subtracted_from_mean)
        return image_thresholed
