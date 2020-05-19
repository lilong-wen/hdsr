import numpy as np
from src.document_binarization import DocumentBinarization
from src.document_normalization import DocumentNormalization


def process_image(image_data, line_height_normalized):
    processed_image = DocumentBinarization.convert_to_grayscale(image_data)
    processed_image = DocumentNormalization.scale_y(processed_image, line_height_normalized)
    processed_image = np.transpose(processed_image, (1, 0))  # swap dimensions
    width = processed_image.shape[0]
    height = processed_image.shape[1]
    processed_image = processed_image.reshape(width, height)
    processed_image = processed_image.astype('float32') / 255  # scale pixel values
    return processed_image


def pad_sequence(sequence, line_height_normalized, line_width_padded):
    output_sequence = sequence
    for i in range(len(sequence), line_width_padded):
        append = np.array([1.0 for i in range(0, line_height_normalized)])
        append = np.reshape(append, (1, line_height_normalized))
        output_sequence = np.append(output_sequence, append, axis=0)
    return output_sequence
