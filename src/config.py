import os


def get_base_dir():
    return os.path.dirname(__file__)


def get_path_to_file_in_assets(*args):
    return os.path.join(get_base_dir(), 'assets', *args)


def get_path_to_image_file_in_assets(*args):
    return os.path.join(get_base_dir(), 'assets', 'images', *args)


def get_path_to_model_dir_in_assets(*args):
    return os.path.join(get_base_dir(), 'assets', 'models', *args)


if __name__ == "__main__":
    pass
