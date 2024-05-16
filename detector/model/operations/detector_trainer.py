import csv
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Union
from matplotlib import pyplot as plt

import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import utilities.base_data_utils as utils
from tqdm import tqdm
import utilities.config as cfg
from utilities.coco_eval import CocoEvaluator
from utilities.coco_utils import get_coco_api_from_dataset
from utilities.early_stopping import EarlyStopping

from detector.data.dataloaders import get_train_val_dataloaders
from detector.model.model import create_mask_r_cnn_on_device


class MaskRCNNTrainer:
    def __init__(
        self,
        data_dir: str,
        settings: SimpleNamespace,
        finetune_path: Union[str, None] = None,
        fix_seed: bool = False,
    ) -> None:
        """
        Initialize the MaskRCNNTrainer class.

        Args:
            data_dir (str): The directory path of the data.
            settings (SimpleNamespace): The settings for the trainer.
            finetune_path (Union[str, None], optional): The path to the pre-trained model weights. Defaults to None.
            fix_seed (bool, optional): Whether to fix the random seed. Defaults to False.
        """
        self.data_dir = data_dir
        self.settings = settings
        self.finetune_path = finetune_path
        self.fix_seed = fix_seed
        self.num_classes = len(settings.class_names) + 1
        self.training_loader, self.validation_loader = get_train_val_dataloaders(
            data_dir, settings, fix_seed=fix_seed
        )
        self.model_device = (
            "cuda" if utils.get_available_device_type() == "cuda" else "cpu"
        )
        self.starting_lr = float(settings.starting_lr)
        self.model = create_mask_r_cnn_on_device(
            self.model_device, self.num_classes, pretrained=not finetune_path
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.starting_lr)
        self.lr_scheduler = None
        self.class_names = settings.class_names
        self.avg_train_losses = []
        self.avg_valid_losses = []
        self.avg_eval_scores = []
        self.per_class_map = MeanAveragePrecision(class_metrics=True)
        self.per_class_seg_map = MeanAveragePrecision(
            class_metrics=True, iou_type="segm"
        )

    def _load_in_weights(
        self, input_path: str, optimizer: bool = False, gpu: bool = True
    ) -> None:
        """
        Load the weights of the model.

        Args:
            input_path (str): The path to the weights file.
            optimizer (bool, optional): Whether to load the optimizer state as well. Defaults to False.
            gpu (bool, optional): Whether to load the weights on GPU. Defaults to True.
        """
        logging.info(f"Loading in weights from saved checkpoint: {input_path}.")
        map_location = self.model_device if gpu else "cpu"
        model_dict = torch.load(input_path, map_location=map_location)
        self.model.load_state_dict(model_dict)
        if optimizer:
            self.optimizer.load_state_dict(model_dict["optimizer_state_dict"])

    def train_model(self, output_path: str, num_epochs: int, patience: int) -> None:
        """
        Train the model.

        Args:
            output_path (str): The path to save the trained model.
            num_epochs (int): The number of epochs to train.
            patience (int): The number of epochs to wait for early stopping.
        """
        self.train_losses = []
        self.valid_losses = []
        if self.finetune_path is None:
            self.model = create_mask_r_cnn_on_device(
                self.model_device, self.num_classes, pretrained=True
            )
        else:
            self._load_in_weights(self.finetune_path, optimizer=False)
        early_stopping = EarlyStopping(
            patience=patience, verbose=True, path=output_path
        )
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.starting_lr,
            steps_per_epoch=len(self.training_loader),
            epochs=num_epochs,
            pct_start=self.settings.pct_lr_inc,
        )
        logging.info(f"Training for {num_epochs} epochs.")
        for epoch in range(num_epochs):
            self.train_one_epoch(epoch, print_freq=20)
            self.calculate_validation_loss()
            self.avg_train_losses.append(np.average(self.train_losses))
            self.avg_valid_losses.append(np.average(self.valid_losses))
            if (epoch % self.settings.evaluate_freq == 0) or (epoch == num_epochs - 1):
                self.evaluate(self.validation_loader)
            early_stopping(
                self.avg_valid_losses[-1],
                self.model,
                self.optimizer,
                self.class_names,
                0,
            )
            if early_stopping.early_stop:
                break

    def calculate_validation_loss(self) -> None:
        """
        Calculate the validation loss.
        """
        logging.info("Calculating validation losses.")
        self.valid_losses = []
        with torch.no_grad():
            for images, targets in tqdm(
                self.validation_loader,
                desc="Validation batch",
                bar_format=cfg.TQDM_BAR_FORMAT,
            ):
                images = [image.to(self.model_device) for image in images]
                targets = [
                    {k: torch.as_tensor(v).to(self.model_device) for k, v in t.items()}
                    for t in targets
                ]
                loss_dict = self.model(images, targets)
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss_value = losses_reduced.item()
                self.valid_losses.append(loss_value)

    def train_one_epoch(self, epoch: int, print_freq: int) -> None:
        """
        Train the model for one epoch.

        Args:
            epoch (int): The current epoch number.
            print_freq (int): The frequency to print the training progress.
        """
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        header = "Epoch: [{}]".format(epoch)
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(self.training_loader) - 1)
            lr_scheduler = utils.warmup_lr_scheduler(
                self.optimizer, warmup_iters, warmup_factor
            )
        else:
            lr_scheduler = self.lr_scheduler
        for images, targets in metric_logger.log_every(
            self.training_loader, print_freq, header
        ):
            images = [image.to(self.model_device) for image in images]
            targets = [
                {k: torch.as_tensor(v).to(self.model_device) for k, v in t.items()}
                for t in targets
            ]
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            self.train_losses.append(loss_value)
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            lr_scheduler.step()
            metric_logger.update(loss=losses, **loss_dict)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

    def evaluate(self, validation_loader) -> CocoEvaluator:
        """
        Evaluate the model.

        Args:
            validation_loader: The data loader for validation dataset.

        Returns:
            CocoEvaluator: The CocoEvaluator object.
        """
        cpu_device = torch.device("cpu")
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"
        coco = get_coco_api_from_dataset(validation_loader.dataset)
        iou_types = utils._get_iou_types(self.model)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        for images, targets in metric_logger.log_every(validation_loader, 100, header):
            images = [img.to(self.model_device) for img in images]
            outputs = self.model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {
                target["image_id"]: output for target, output in zip(targets, outputs)
            }
            coco_evaluator.update(res)
        metric_logger.synchronize_between_processes()
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        return coco_evaluator

    def output_loss_fig(self, model_out_path: Path) -> None:
        """
        Output the loss figure.

        Args:
            model_out_path (Path): The path to save the loss figure.
        """
        fig = plt.figure(figsize=(10, 8))
        plt.plot(
            range(1, len(self.avg_train_losses) + 1),
            self.avg_train_losses,
            label="Training Loss",
        )
        plt.plot(
            range(1, len(self.avg_valid_losses) + 1),
            self.avg_valid_losses,
            label="Validation Loss",
        )
        minposs = self.avg_valid_losses.index(min(self.avg_valid_losses)) + 1
        plt.axvline(
            minposs, linestyle="--", color="r", label="Early Stopping Checkpoint"
        )
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.xlim(0, len(self.avg_train_losses) + 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        output_dir = model_out_path.parent
        fig_out_pth = output_dir / f"{model_out_path.stem}_loss_plot.png"
        fig.savefig(fig_out_pth, bbox_inches="tight")
        epoch_lst = range(len(self.avg_train_losses))
        rows = zip(epoch_lst, self.avg_train_losses, self.avg_valid_losses)
        with open(output_dir / f"{model_out_path.stem}_train_stats.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(("Epoch", "Train Loss", "Valid Loss"))
            writer.writerows(rows)
