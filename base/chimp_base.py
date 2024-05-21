from base.chimp_errors import InputError

class ChimpBase():

    def __init__(self, model_path, image_list):
        self.model_path = model_path
        self.image_list = image_list
        self.check_image_list()

    def check_image_list(self):
        """Check image list type.

        Raises:
            InputError: If image list is not a list.
        """
        if not isinstance(self.image_list, list):
            raise InputError(self.image_list, "Object passsed in as image list is not a list!")

    def check_model_path(self):
        """Check validity of path given for model file.

        Raises:
            InputError: If path is not a file.
        """
        if not self.model_path.is_file():
            raise InputError(self.model_path, "The path given is not a file.")
       
    def load_model(self):
        """Loads the model file and links it to the images in the image directory.
        """
        raise NotImplementedError
