import os

from src.image_container import ImageContainer


class FileLoader:
    @staticmethod
    def load_image(path):
        if FileLoader._is_file(path):
            image_container = FileLoader._read_image_from_system(path)
            if FileLoader._is_valid_image_type(image_container):
                return image_container
            else:
                raise IOError
        else:
            raise IOError

    @staticmethod
    def save_image(image_container, output_folder):
        FileLoader._create_output_folder(output_folder)
        image_path = image_container.save_processed_image(output_folder)
        return image_path

    @staticmethod
    def load_text_file(path):
        if FileLoader._is_file(path):
            data_file = open(path, 'r')
            training_data_list = data_file.readlines()
            data_file.close()
            return training_data_list
        else:
            print("Could not find path: ", path)
            raise OSError

    @staticmethod
    def _create_output_folder(output_folder):
        if not FileLoader._does_folder_exists(output_folder):
            os.makedirs(output_folder)

    @staticmethod
    def _read_image_from_system(path):
        return ImageContainer(path)

    @staticmethod
    def _is_valid_image_type(image_container):
        if image_container.get_unprocessed_image() is None:
            return False
        else:
            return True

    @staticmethod
    def _does_folder_exists(path):
        return os.path.isdir(path)

    @staticmethod
    def _is_file(path):
        return os.path.isfile(path)
