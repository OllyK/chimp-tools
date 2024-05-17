from pathlib import Path
import logging
import pandas as pd
from tqdm import tqdm
from classifier.model.model import create_model_on_device
from classifier.data.dataloaders import get_prediction_dataloader
import torch
from collections import defaultdict

import utilities.config as cfg
import torch.nn as nn


class ImageFolderClassifier:
    """
    A class for predicting image classes using a trained model.

    Args:
        model_path (Path): Path to the trained model.
        image_list (list): List of image paths to predict classes for.
        img_size (int): Size of the input images.
        device (str, optional): Device to run the prediction on (default is "0").

    Attributes:
        model_path (Path): Path to the trained model.
        image_list (list): List of image paths to predict classes for.
        img_size (int): Size of the input images.
        device (str): Device to run the prediction on.
        output_dict (defaultdict): Dictionary to store the prediction results.
        model_struc_dict (dict): Dictionary containing the model structure.
        model (nn.Module): Trained model.
        class_names (list): List of class names.
        idx_to_class (dict): Mapping of class indices to class names.
        output_dir (Path): Output directory path.
        class_csv_output_path (Path): Path to the output CSV file.

    """

    def __init__(
        self, model_path: Path, image_list: list, img_size: int, device: str = "0"
    ):
        self.model_path = model_path
        self.image_list = image_list
        self.img_size = img_size
        self.device = device
        self.output_dict = defaultdict(list)
        self.model_struc_dict = None
        self.model = None
        self.class_names = None
        self.idx_to_class = None
        self.output_dir = None
        self.class_csv_output_path = None

        self.torch_device = self._get_torch_device(device)
        logging.info("Creating prediction dataloader")
        self.dataloader = get_prediction_dataloader(
            image_list, img_size=img_size, batch_size=cfg.BIG_CUDA_PRED_BATCH
        )
        self._load_in_model(model_path)

    def _load_in_model(self, model_path):
        """
        Load the trained model from the given path.

        Args:
            model_path (Path): Path to the trained model.

        """
        logging.info(f"Loading in model from {model_path}")
        model_dict = self._load_model_dict(model_path)
        self.model_struc_dict = model_dict["model_struc_dict"]
        self.model = create_model_on_device(
            self.device, self.model_struc_dict, pretrained=False
        )
        self.model.load_state_dict(model_dict["model_state_dict"])
        self.model.eval()
        self.class_names = model_dict["class_names"]
        logging.info(f"Class names are {self.class_names}")
        self.idx_to_class = {i: j for i, j in enumerate(self.class_names)}
        del model_dict

    def run(self, output_dir):
        """
        Run the image classification and output the results.

        Args:
            output_dir (Path): Output directory path.

        """
        self._make_output_dir(output_dir)
        self._predict_classes()
        self._output_csv()

    def _make_output_dir(self, output_dir):
        """
        Make a subdirectory in the specified path for output and set paths for CSV output.

        Args:
            output_dir (Path): Output directory path.

        """
        self.output_dir = output_dir / "chimp_output"
        self.class_csv_output_path = self.output_dir / "chimp_results.csv"
        logging.info(f"Making directory for output {self.output_dir}")
        self.output_dir.mkdir(exist_ok=True)

    def _load_model_dict(self, model_path):
        """
        Load the model dictionary from the given path.

        Args:
            model_path (Path): Path to the trained model.

        Returns:
            dict: Model dictionary.

        """
        if self.device == "cpu":
            map_location = self.device
        else:
            map_location = f"cuda:{self.device}"
        return torch.load(model_path, map_location=map_location)

    def _get_torch_device(self, device):
        """
        Get the torch device based on the specified device string.

        Args:
            device (str): Device string.

        Returns:
            torch.device: Torch device.

        """
        if device == "cpu":
            device_str = device
        else:
            device_str = "cuda:" + str(device)
        return torch.device(device_str)

    def _predict_classes(self):
        """
        Predict the image classes and store the outputs in a dictionary.

        Returns:
            dict: Dictionary containing the prediction results.

        """
        s_max = nn.Softmax(dim=1)
        logging.info("Predicting image classes.")
        with torch.no_grad():
            for batch in tqdm(
                self.dataloader,
                desc="Prediction batch",
                bar_format=cfg.TQDM_BAR_FORMAT,
            ):

                output = self.model(batch["image"].to(self.torch_device))
                probs = s_max(output)
                max_probs = torch.max(probs, dim=1)
                pred_class_name = [
                    self.idx_to_class[idx] for idx in max_probs.indices.tolist()
                ]
                # Append results to lists in the output dictionary
                for i, filename in enumerate(batch["filename"]):
                    self.output_dict["path"].append(filename)
                    self.output_dict["pred_class"].append(pred_class_name[i])
                    self.output_dict["pred_max_prob"].append(
                        max_probs.values.tolist()[i]
                    )
                    for class_name, prob in zip(self.class_names, probs.tolist()[i]):
                        self.output_dict[class_name].append(prob)
        return dict(self.output_dict)

    def _output_csv(self):
        """
        Convert the data dictionary into a CSV file for output and save it to disk.

        """
        chimp_table = pd.DataFrame(data=self.output_dict)
        chimp_table.to_csv(self.class_csv_output_path, index=False)
        logging.info("Results output to: {}".format(self.class_csv_output_path))
