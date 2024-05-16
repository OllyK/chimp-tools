import csv
import logging
import math
import sys
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
    """Class that provides methods to train a Mask-R-CNN network"""

    def __init__(
        self,
        data_dir: Path,
        settings: SimpleNamespace,
        finetune_path: Union[Path, None] = None,
        fix_seed: bool = False,
    ):
        self.lr_scheduler = None
        self.avg_train_losses = []  # per epoch training loss
        self.avg_valid_losses = []  #  per epoch validation loss
        self.avg_eval_scores = []  #  per epoch evaluation score
        self.data_dir = data_dir
        self.settings = settings
        self.finetune_path = finetune_path
        self.fix_seed = fix_seed
        self.class_names = settings.class_names
        self.num_classes = len(self.class_names)
        self.training_loader, self.validation_loader = get_train_val_dataloaders(
            data_dir, settings, fix_seed=fix_seed
        )
        # Params for model training
        self.detector_method = utils.get_detector_method(settings)
        device_type = utils.get_available_device_type()
        if device_type == "cuda":
            self.model_device = f"cuda:{int(settings.cuda_device)}"
        else:
            self.model_device = device_type
        self.patience = settings.patience
        self.starting_lr = float(settings.starting_lr)
        self.per_class_map = MeanAveragePrecision(class_metrics=True)
        self.per_class_seg_map = MeanAveragePrecision(
            class_metrics=True, iou_type="segm"
        )

    def _create_model_and_optimiser(self, learning_rate, pretrained=False):
        logging.info(f"Setting up the model on device {self.model_device}.")
        self.model = create_mask_r_cnn_on_device(
            self.model_device, self.num_classes, pretrained=pretrained
        )
        self.optimizer = self._create_optimizer(learning_rate)

    def _create_optimizer(self, learning_rate):
        return torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def _create_oc_lr_scheduler(self, num_epochs, lr_to_use):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr_to_use,
            steps_per_epoch=len(self.training_loader),
            epochs=num_epochs,
            pct_start=self.settings.pct_lr_inc,
        )

    def _load_in_model_and_optimizer(self, learning_rate, input_path, optimizer=False):
        self._create_model_and_optimiser(learning_rate)
        logging.info(f"Loading in weights from saved checkpoint: {input_path}.")
        self._load_in_weights(input_path, optimizer=optimizer)
    
    def _load_in_weights(self, input_path, optimizer=False, gpu=True):
        # load the last checkpoint with the best model
        if gpu:
            map_location = self.model_device
        else:
            map_location = "cpu"
        model_dict = torch.load(input_path, map_location=map_location)
        logging.info("Loading model weights.")
        self.model.load_state_dict(model_dict)
        if optimizer:
            logging.info("Loading optimizer weights.")
            self.optimizer.load_state_dict(model_dict["optimizer_state_dict"])

    def _create_early_stopping(
        self,
        output_path,
        patience,
        best_score=None,
    ):
        return EarlyStopping(
            patience=patience,
            verbose=True,
            path=output_path,
            best_score=best_score,
        )

    def train_model(
        self,
        output_path: Path,
        num_epochs: int,
        patience: int,
    ):
        self.train_losses = []
        self.valid_losses = []
        if self.detector_method == utils.DetectorMethod.STANDARD:
            if self.finetune_path is None:
                self._create_model_and_optimiser(self.starting_lr, pretrained=True)
            else:
                self._load_in_model_and_optimizer(self.starting_lr, self.finetune_path)
            early_stopping = self._create_early_stopping(
                output_path,
                patience,
            )
            self.lr_scheduler = self._create_oc_lr_scheduler(
                num_epochs, self.starting_lr
            )
            logging.info(f"Training for {num_epochs} epochs.")
            for epoch in range(num_epochs):
                self.train_one_epoch(epoch, print_freq=20)
                # Evaluate on validation set
                self.calculate_validation_loss()
                # if epoch == num_epochs - 1:  # altered to speed up training
                #     self.evaluate(self.validation_loader)
                self.avg_train_losses.append(np.average(self.train_losses))
                self.avg_valid_losses.append(np.average(self.valid_losses))
                logging.info(
                    f"Epoch {epoch}. Training loss: {self.avg_train_losses[-1]}, Validation Loss: "
                    f"{self.avg_valid_losses[-1]}."
                )
                # torch.cuda.empty_cache()
                if (epoch % self.settings.evaluate_freq == 0) or (epoch == num_epochs -1):  # altered to speed up training
                    self.evaluate(self.validation_loader)
                #self.evaluate_using_torchmetrics()
                 # early_stopping needs the validation loss to check if it has decreased,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(
                    self.avg_valid_losses[-1],
                    self.model,
                    self.optimizer,
                    self.class_names,
                    0,
                )

                if early_stopping.early_stop:
                    logging.info("Early stopping")
                    break
            # logging.info(f"Saving model to {output_path}")
            # torch.save(self.model, output_path)

    def calculate_validation_loss(self):
        logging.info("Calculating validation losses.")
        with torch.no_grad():
            for images, targets in tqdm(
                self.validation_loader,
                desc="Validation batch",
                bar_format=cfg.TQDM_BAR_FORMAT,
            ):
                images = list(image.to(self.model_device) for image in images)
                targets = [
                    {k: torch.as_tensor(v).to(self.model_device) for k, v in t.items()} for t in targets
                ]
                loss_dict = self.model(images, targets)
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss_value = losses_reduced.item()
                self.valid_losses.append(loss_value)

    def train_one_epoch(self, epoch, print_freq):
        """Adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html"""
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        header = "Epoch: [{}]".format(epoch)

        lr_scheduler = self.lr_scheduler
        # if epoch == 0:
        #     logging.info("Warmup learning rate scheduler.")
        #     warmup_factor = 1.0 / 1000
        #     warmup_iters = min(1000, len(self.training_loader) - 1)

        #     lr_scheduler = utils.warmup_lr_scheduler(
        #         self.optimizer, warmup_iters, warmup_factor
        #     )
        # else:
        #     lr_scheduler = self.lr_scheduler
        for images, targets in metric_logger.log_every(
            self.training_loader, print_freq, header
        ):
            images = list(image.to(self.model_device) for image in images)
            targets = [{k: torch.as_tensor(v).to(self.model_device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = utils.reduce_dict(loss_dict)
            # losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses.item()
            self.train_losses.append(loss_value)

            # if not math.isfinite(loss_value):
            #     print("Loss is {}, stopping training".format(loss_value))
            #     print(loss_dict_reduced)
            #     sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            lr_scheduler.step()

            metric_logger.update(loss=losses, **loss_dict)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

    def evaluate_using_torchmetrics(self):
        logging.info("Calculating MAP Metrics.")
        self.model.eval()
        for images, targets in tqdm(
                self.validation_loader,
                desc="Metrics batch",
                bar_format=cfg.TQDM_BAR_FORMAT,
            ):
                images = list(image.to(self.model_device) for image in images)
                targets = [{k: v.to(self.model_device) for k, v in t.items()} for t in targets]
                preds = self.model(images)
                # for result in preds:
                #     result["masks"] = (result["masks"] > 155).byte()
                self.per_class_map.update(preds, targets)
                # self.per_class_seg_map.update(preds, targets)
        results = self.per_class_map.compute()
        logging.info(f"Computed metrics: {results}.")
        self.avg_eval_scores.append(results["map"].item())
        self.per_class_map.reset()
        # self.per_class_seg_map.reset()

    @torch.no_grad()
    def evaluate(self, validation_loader):
        # n_threads = torch.get_num_threads()
        # # FIXME remove this and make paste_masks_in_image run on the GPU
        # torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"

        coco = get_coco_api_from_dataset(validation_loader.dataset)
        iou_types = utils._get_iou_types(self.model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for images, targets in metric_logger.log_every(validation_loader, 100, header):
            images = list(img.to(self.model_device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = self.model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"]: output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

         # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        # torch.set_num_threads(n_threads)
        return coco_evaluator

    def output_loss_fig(self, model_out_path: Path) -> None:
        """Save out a figure showing training and validation loss versus
        epoch number.
        Args:
            model_out_path (Path): Path to the model output by the trainer.
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

        minposs = (
            self.avg_valid_losses.index(min(self.avg_valid_losses)) + 1
        )  # find position of lowest validation loss
        plt.axvline(
            minposs, linestyle="--", color="r", label="Early Stopping Checkpoint"
        )

        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.xlim(0, len(self.avg_train_losses) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        output_dir = model_out_path.parent
        fig_out_pth = output_dir / f"{model_out_path.stem}_loss_plot.png"
        logging.info(f"Saving figure of training/validation losses to {fig_out_pth}")
        fig.savefig(fig_out_pth, bbox_inches="tight")
        # Output a list of training stats
        epoch_lst = range(len(self.avg_train_losses))
        rows = zip(
            epoch_lst,
            self.avg_train_losses,
            self.avg_valid_losses,
            #self.avg_eval_scores,
        )
        with open(output_dir / f"{model_out_path.stem}_train_stats.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(("Epoch", "Train Loss", "Valid Loss"))
            for row in rows:
                writer.writerow(row)
                