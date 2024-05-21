from base.chimp_errors import InputError

def check_image_dir_path(image_dir_path):
    """Check validity of path given for image directory.

    Raises:
        InputError: If path is not a directory.
        InputError: If directory is empty.
    """
    if not image_dir_path.is_dir():
        raise InputError(image_dir_path, "The path given is not a directory.")
    if not any(image_dir_path.iterdir()):
        raise InputError(image_dir_path, "The directory is empty.")

def check_image_list_file(list_file_path):
    """Check validity of path given for an image list file.

    Raises:
        InputError: If path is not a directory.
        InputError: If wrong filetype.
    """
    if not list_file_path.is_file():
        raise InputError(list_file_path, "The path given is not a file.")
    if list_file_path.suffix not in [".txt", ".TXT"]:
        raise InputError(list_file_path, "The path given is not a text file.")

def check_ispyb_config_file(ispyb_config):
    """Check validity of path given for an image list file.

    Raises:
        InputError: If path is not a directory.
        InputError: If wrong filetype.
    """
    if not ispyb_config.is_file():
        raise InputError(ispyb_config, "The path given is not a file.")
    if ispyb_config.suffix not in [".cnf", ".cfg"]:
        raise InputError(ispyb_config, "The path given is not a config file.")
