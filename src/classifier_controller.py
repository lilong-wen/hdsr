from keras import backend as KerasBackend
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from src.file_loader import FileLoader
from src.setup_trainings_data import SetupTrainingsData
from src.reccurent_neural_network import RecurrentNeuralNetwork


class Classifier:
    DATASET_TYP = {
        'cvl': 2,
        'no_train': 3,
        'iam': 1
    }

    def __init__(self):
        self.file_loader = FileLoader()
        self.classifier_model = None
        self.classifier_type = None

        KerasBackend.set_image_data_format('channels_last')

    def load_dataset(self, dataset=DATASET_TYP['cvl']):
        database = None
        if dataset == Classifier.DATASET_TYP['iam']:
            database = SetupTrainingsData.get_database_by_name("iam")
        elif dataset == Classifier.DATASET_TYP['cvl']:
            database = SetupTrainingsData.get_database_by_name("cvl")
        elif dataset == Classifier.DATASET_TYP['no_train']:
            database = SetupTrainingsData.get_database_by_name("no_train")
        else:
            database = SetupTrainingsData.get_database_by_name(dataset)

        if database:
            return database.get_dataset()
        else:
            return None

    def define_model(self, **kwargs):
        self.classifier_model = RecurrentNeuralNetwork(charset=kwargs['charset'],
                                                       number_of_memory_units=kwargs['memory_units'],
                                                       pool_size=kwargs['pool_size'],
                                                       line_width_padded=kwargs['line_width_padded'],
                                                       line_height_normalized=kwargs['line_height_normalized'],
                                                       max_string_length=kwargs['max_string_length'])
        return self.classifier_model

    def load_weights(self, path):
        if self.classifier_model is not None:
            model = self.classifier_model.get_model()
            model.load_weights(path)  # TODO: is it necessary to set the model variable in the model class

    def train(self, generator, epochs, callbacks=None, visualize=True):
        if generator:
            history = self.classifier_model.train(generator, epochs, callbacks)
            if visualize:
                history_dict = history.history
                loss_values = history_dict['loss']
                val_loss_values = history_dict['val_loss']

                epochs = range(1, len(loss_values) + 1)

                plt.plot(epochs, loss_values, 'bo', label='Verlust Training')
                plt.plot(epochs, val_loss_values, 'b', label='Verlust Validierung')
                plt.title('Wert der Verlustfunktion Training/Validierung')
                plt.xlabel('Epochen')
                plt.ylabel('Wert der Verlustfunktion')
                plt.legend()

                """ accuracy can't be computed """
                plt.show()
