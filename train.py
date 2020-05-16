import os
import sys
from sacred import Experiment

from src.classifier_controller import Classifier
from src.data_generator import DataGenerator
from src.visualization_callback import VisualizationCallback
from src.no_train_callback import NOTRAINCallback
from src.config import get_path_to_model_dir_in_assets

ex_images = Experiment()


@ex_images.config
def default_config():
    batch_size = 32
    epochs = 2
    memory_units = 512
    pool_size = 2

    charset = 67  # a-zA-Z0-9,.- /
    line_width_padded = 535
    line_height_normalized = 64
    #max_string_length = 7
    max_string_length = 8

    #force_new = False
    force_new = True

    training_samples = None


@ex_images.main
def run(training_samples, batch_size, epochs, memory_units, pool_size, charset, line_width_padded, line_height_normalized,
        max_string_length, force_new):

    classifier = Classifier()

    image_paths = None
    image_labels = None

    if training_samples is None:
        image_paths, image_labels = classifier.load_dataset(Classifier.DATASET_TYP['cvl'])
    else:
        image_paths, image_labels = classifier.load_dataset(training_samples)

    # for choicing model
    '''
    no_train_image_paths, no_train_image_labels = classifier.load_dataset(Classifier.DATASET_TYP['no_train'])
    generator_no_train = DataGenerator(downsample_factor=(pool_size ** 2),
                              line_width_padded=line_width_padded,
                              line_height_normalized=line_height_normalized, max_string_length=max_string_length)
    generator_no_train.setup_training(image_paths=no_train_image_paths, image_labels=no_train_image_labels, batch_size=1,
                                      validation_split=0.0)
    '''

    generator = DataGenerator(downsample_factor=(pool_size ** 2),
                              line_width_padded=line_width_padded,
                              line_height_normalized=line_height_normalized, max_string_length=max_string_length)
    generator.setup_training(image_paths=image_paths, image_labels=image_labels, batch_size=batch_size)

    model = classifier.define_model(charset=charset,
                                    line_height_normalized=line_height_normalized,
                                    memory_units=memory_units, pool_size=pool_size,
                                    line_width_padded=line_width_padded, max_string_length=max_string_length)

    if force_new is False:
        rnn_model = model.get_model()
        rnn_model.load_weights(get_path_to_model_dir_in_assets('rnn_prediction_2020_04_10_01_20_07.h5'))

    test_function = model.get_test_function()
    visualization_callback = VisualizationCallback(test_function, generator.generate_validation())
    #no_train_callback = NOTRAINCallback(test_function, generator_no_train.generate_prediction())

    #classifier.train(generator, epochs, callbacks=[generator, visualization_callback], visualize=True)
    classifier.train(generator, epochs, callbacks=[generator, visualization_callback], visualize=True)


if __name__ == '__main__':
    head, tail = os.path.split(os.path.join(os.path.abspath(__file__)))
    PACKAGE_DIR = os.path.join(head, '..{}..{}'.format(os.sep, os.sep))

    sys.path.insert(0, PACKAGE_DIR)

    ex_images.run_commandline()
