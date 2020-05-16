import re
import os
from glob import glob

from src.config import get_path_to_file_in_assets
from src.file_loader import FileLoader


class SetupTrainingsData(object):
    def __init__(self):
        pass

    @staticmethod
    def get_database_by_name(database_name):
        if database_name == "iam":
            return IamDatabase()
        elif database_name == "cvl":
            return CvlDatabase()
        elif database_name == "no-train":
            return NoTrainDatabase()
        else:
            return TextLines(database_name)


class IamDatabase(SetupTrainingsData):
    def __init__(self, path_to_images=get_path_to_file_in_assets("iam_dataset\\lines\\"),
                 path_to_labels=get_path_to_file_in_assets("iam_dataset\\ascii\\lines.txt")):
        self._file_pointer = 0
        self.path_to_line_images = path_to_images
        self.path_to_line_transcriptions = path_to_labels
        self.image_file_extension = ".png"

        self.line_transcriptions = []

        self._read_files()

    @property
    def file_pointer(self):
        return self._file_pointer

    def _read_files(self):
        self.line_transcriptions = FileLoader.load_text_file(self.path_to_line_transcriptions)

    def get_dataset(self):
        image_paths = []
        image_labels = []
        while self._file_pointer <= len(self.line_transcriptions):
            text_line = self.line_transcriptions[self._file_pointer]
            self._increment_pointer()
            if self._is_comment(text_line):
                continue
            else:
                segmentation_result = self._extract_segmentation_result(text_line)
                if segmentation_result.lower() == "ok":
                    image_path = self._extract_image_path(text_line)
                    image_paths.append(image_path)

                    transcription = self._extract_transcription(text_line)
                    image_label = self._get_transcription(transcription)
                    image_labels.append(image_label)
                else:
                    continue
        return image_paths, image_labels

    def _is_comment(self, line):
        return line[:1] is "#"

    def _increment_pointer(self):
        self._file_pointer = self._file_pointer + 1

    def _extract_image_path(self, line):
        path_position = 0
        file_name = re.split("\s+", line)[path_position]
        path_dirs = file_name.split('-')
        image_path = path_dirs[0] + '\\' + path_dirs[0] + '-' + path_dirs[
            1] + '\\' + file_name + self.image_file_extension
        return image_path

    def _extract_transcription(self, line):
        transcription_position = 8
        return re.split("\s+", line)[transcription_position]

    def _extract_segmentation_result(self, line):
        segmentation_result_position = 1
        return re.split("\s+", line)[segmentation_result_position]

    def _get_transcription(self, r_transcription):
        return r_transcription.replace('|', ' ')


class CvlDatabase(SetupTrainingsData):
    def __init__(self, path_to_images=get_path_to_file_in_assets("cvl_dataset\\train")):
        self.path_to_images = path_to_images

    def get_dataset(self):
        image_paths = glob(self.path_to_images + '\\*.png')
        image_labels = []
        for image_path in image_paths:
            head, tail = os.path.split(image_path)
            image_labels.append(tail[:tail.find('-')])
        return image_paths, image_labels


class NoTrainDatabase(SetupTrainingsData):
    def __init__(self, path_to_images=get_path_to_file_in_assets("no_train_dataset\\no_train")):
        self.path_to_images = path_to_images

    def get_dataset(self):
        image_paths = glob(self.path_to_images + '\\*.png')
        image_labels = []
        for image_path in image_paths:
            head, tail = os.path.split(image_path)
            image_labels.append(tail[:tail.find('-')])
        return image_paths, image_labels



class TextLines(SetupTrainingsData):
    def __init__(self, path_to_training_samples):
        self.path_to_images = path_to_training_samples
        self.path_to_labels = path_to_training_samples

        self.image_file_extension = '.png'

    def get_dataset(self):
        image_paths = glob(os.path.join(self.path_to_images, '*.{}'.format(self.image_file_extension)))
        label_paths = glob(os.path.join(self.path_to_labels + '*.txt'))

        assert len(image_paths) == len(label_paths)

        labels = []

        for label_path in label_paths:
            file = open(label_path, 'r', encoding='utf-8')
            labels.append(file.read())

        return image_paths, labels
