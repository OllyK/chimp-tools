import csv
import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import numpy as np
import pandas as pd
import torch
import torchmetrics
from collections import OrderedDict
import utilities.config as cfg
from classifier.data.dataloaders import get_train_val_dataloaders
from classifier.model.model import create_model_on_device
from utilities.early_stopping import EarlyStopping
from matplotlib import pyplot as plt
from tqdm import tqdm

class ImageClassifierTrainer:
    """Class that provides methods to train a deep learning image
    classifier.
    """

    def __init__(
        self,
        training_data: pd.DataFrame,
        settings: SimpleNamespace,
        finetune_path: Union[Path, None] = None,
        imbalanced: bool = False,
        fix_seed: bool = False,
        marco_validation_data: Union[pd.DataFrame, None] = None,
        distributed: bool = False
    ):
        """Inits ImageClassifierTrainer.
        Args:
            training_data (pd.DataFrame): A Pandas dataframe with columns for image path and class label
            settings (SimpleNamespace): A training settings object.
        """
        self.training_data = training_data
        self.settings = settings
        self.finetune_path = finetune_path
        self.imbalanced = imbalanced
        self.fix_seed = fix_seed
        self.marco_validation_data = marco_validation_data
        self.distributed = distributed
        self.class_names = settings.class_names
        self.training_loader, self.validation_loader = get_train_val_dataloaders(
            training_data, settings, imbalanced=imbalanced, fix_seed=fix_seed, validation_data=marco_validation_data
        )
        # Params for learning rate finder
        self.starting_lr = float(settings.starting_lr)
        self.end_lr = float(settings.end_lr)
        self.log_lr_ratio = self._calculate_log_lr_ratio()
        self.lr_find_epochs = settings.lr_find_epochs
        self.lr_reduce_factor = settings.lr_reduce_factor
        # Params for model training
        self.model_device_num = int(settings.cuda_device)
        self.patience = settings.patience
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.num_classes = len(settings.class_names)
        self.per_class_accuracy = torchmetrics.Accuracy(
            num_classes=self.num_classes, average=None
        ).to(self.model_device_num)
        self.accuracy = torchmetrics.Accuracy().to(self.model_device_num)
        if finetune_path is None:
            self.model_struc_dict = self._get_model_struc_dict(settings)
        else:
            self.model_struc_dict = self._load_model_struc_dict(finetune_path)
        self.avg_train_losses = []  # per epoch training loss
        self.avg_valid_losses = []  #  per epoch validation loss
        self.avg_eval_scores = []  #  per epoch evaluation score

    def _load_model_struc_dict(self, finetune_path, gpu=True):
        model_dict = torch.load(finetune_path)
        return model_dict["model_struc_dict"]

    def _get_model_struc_dict(self, settings):
        class_names = settings.class_names
        model_struc_dict = settings.model
        model_struc_dict["classes"] = class_names
        model_struc_dict["num_classes"] = len(class_names)
        return model_struc_dict

    def _calculate_log_lr_ratio(self):
        return math.log(self.end_lr / self.starting_lr)

    def _create_optimizer(self, learning_rate):
        return torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def _create_exponential_lr_scheduler(self):
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_exp_stepper)

    def _create_oc_lr_scheduler(self, num_epochs, lr_to_use):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr_to_use,
            steps_per_epoch=len(self.training_loader),
            epochs=num_epochs,
            pct_start=self.settings.pct_lr_inc,
        )

    def _create_early_stopping(
        self, output_path, patience, best_score=None, use_accuracy=False
    ):
        return EarlyStopping(
            patience=patience,
            verbose=True,
            path=output_path,
            model_dict=self.model_struc_dict,
            best_score=best_score,
            use_accuracy=use_accuracy,
            distributed=self.distributed
        )

    def _create_model_and_optimiser(self, learning_rate, frozen=True, pretrained=True):
        logging.info(f"Setting up the model on device {self.model_device_num}.")
        self.model = create_model_on_device(
            self.model_device_num, self.model_struc_dict, pretrained=pretrained, distributed=self.distributed
        )
        if frozen:
            self._freeze_model()
        logging.info(
            f"Model has {self._count_trainable_parameters()} trainable parameters, {self._count_parameters()} total parameters."
        )
        self.optimizer = self._create_optimizer(learning_rate)

    def _load_in_model_and_optimizer(
        self, learning_rate, output_path, frozen=False, optimizer=False
    ):
        self._create_model_and_optimiser(learning_rate, frozen=frozen, pretrained=False)
        logging.info(f"Loading in weights from saved checkpoint: {output_path}.")
        loss_val = self._load_in_weights(output_path, optimizer=optimizer)
        return loss_val

    def _load_in_weights(self, output_path, optimizer=False, gpu=True):
        # load the last checkpoint with the best model
        if gpu:
            map_location = f"cuda:{self.model_device_num}"
        else:
            map_location = "cpu"
        model_dict = torch.load(output_path, map_location=map_location)
        logging.info("Loading model weights.")
        state_dict = model_dict["model_state_dict"]
        if self.distributed:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "module" not in k:
                    k = "module." + k
                else:
                    k = k.replace("features.module", "module.features")
                new_state_dict[k] = v
            state_dict = new_state_dict
        self.model.load_state_dict(state_dict)
        if optimizer:
            logging.info("Loading optimizer weights.")
            self.optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        return model_dict.get("loss_val", np.inf)

    def _count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _freeze_model(self):
        logging.info(
            f"Freezing model with {self._count_trainable_parameters()} trainable parameters, {self._count_parameters()} total parameters."
        )
        self.model.requires_grad_(False)
        try:
            self.model.head.requires_grad_(True)
        except AttributeError:
            self.model.fc.requires_grad_(True)

    def _unfreeze_model(self):
        logging.info(
            f"Unfreezing model with {self._count_trainable_parameters()} trainable parameters, {self._count_parameters()} total parameters."
        )
        self.model.requires_grad_(True)

    def train_model(
        self,
        output_path: Path,
        num_epochs: int,
        patience: int,
        create: bool = True,
        frozen: bool = False,
        second_round: bool = False,
        reload_optimizer: bool = False,
        use_accuracy: bool = False,
    ) -> None:
        """Performs training of model for a number of epochs
        with a learning rate that is determined automatically.
        Args:
            output_path (Path): Path to save model file to.
            num_epochs (int): Number of epochs to train the model for.
            patience (int): Number of epochs to wait for while validation loss is not improving before terminating.
            create (bool, optional): Whether to create a new model and optimizer from scratch. Defaults to True.
            frozen (bool, optional): Whether to freeze the weights for convolutional layers in the encoder. Defaults to False.
        """
        train_losses = []
        valid_losses = []
        if create:
            self._create_model_and_optimiser(self.starting_lr, frozen=frozen)
            lr_to_use = self._run_lr_finder()
            # Recreate model and start training
            self._create_model_and_optimiser(lr_to_use, frozen=frozen)
            early_stopping = self._create_early_stopping(
                output_path, patience, use_accuracy=use_accuracy
            )
        else:
            # Reduce starting LR, since model already partially trained
            self.starting_lr /= self.lr_reduce_factor
            self.end_lr /= self.lr_reduce_factor
            self.log_lr_ratio = self._calculate_log_lr_ratio()
            if self.finetune_path is not None and not second_round:
                input_path = self.finetune_path
            else:
                input_path = output_path
            self._load_in_model_and_optimizer(
                self.starting_lr, input_path, frozen=frozen, optimizer=reload_optimizer
            )
            lr_to_use = self._run_lr_finder()
            min_loss = self._load_in_model_and_optimizer(
                lr_to_use, input_path, frozen=frozen, optimizer=reload_optimizer
            )
            early_stopping = self._create_early_stopping(
                output_path, patience, best_score=-min_loss, use_accuracy=use_accuracy
            )

        # Initialise the One Cycle learning rate scheduler
        lr_scheduler = self._create_oc_lr_scheduler(num_epochs, lr_to_use)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            tic = time.perf_counter()
            logging.info(f"Epoch {epoch} of {num_epochs}")
            for batch in tqdm(
                self.training_loader,
                desc="Training batch",
                bar_format=cfg.TQDM_BAR_FORMAT,
            ):
                loss = self._train_one_batch(lr_scheduler, batch)
                train_losses.append(loss.item())  # record training loss

            self.model.eval()  # prep model for evaluation
            with torch.no_grad():
                for batch in tqdm(
                    self.validation_loader,
                    desc="Validation batch",
                    bar_format=cfg.TQDM_BAR_FORMAT,
                ):
                    inputs, targets = batch[0].to(self.model_device_num), batch[1].to(
                        self.model_device_num
                    )
                    output = self.model(inputs)  # Forward pass
                    # calculate the loss
                    loss = self.loss_criterion(output, targets)
                    valid_losses.append(loss.item())  # record validation loss
                    self.per_class_accuracy(output, targets)
                    self.accuracy(output, targets)

            toc = time.perf_counter()
            # calculate average loss/metric over an epoch
            self.avg_train_losses.append(np.average(train_losses))
            self.avg_valid_losses.append(np.average(valid_losses))
            accuracy = self.accuracy.compute()
            self.avg_eval_scores.append(accuracy)
            per_class_accuracy = self.per_class_accuracy.compute()
            per_class_output = list(
                zip(self.settings.class_names, per_class_accuracy.tolist())
            )
            logging.info(
                f"Epoch {epoch}. Training loss: {self.avg_train_losses[-1]}, Validation Loss: "
                f"{self.avg_valid_losses[-1]}. Accuracy: {accuracy}, Per class accuracy: {per_class_output}"
            )
            logging.info(f"Time taken for epoch {epoch}: {toc - tic:0.2f} seconds")
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            self.accuracy.reset()
            self.per_class_accuracy.reset()

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(
                self.avg_valid_losses[-1],
                self.model,
                self.optimizer,
                self.class_names,
                self.avg_eval_scores[-1],
            )

            if early_stopping.early_stop:
                logging.info("Early stopping")
                break
        # load the last checkpoint with the best model
        # self._load_in_weights(output_path, optimizer=True)

    def _run_lr_finder(self):
        logging.info("Finding learning rate for model.")
        lr_scheduler = self._create_exponential_lr_scheduler()
        lr_find_loss, lr_find_lr = self._lr_finder(lr_scheduler)
        lr_to_use = self._find_lr_from_graph(lr_find_loss, lr_find_lr)
        logging.info(f"LR to use {lr_to_use}")
        return lr_to_use

    def _train_one_batch(self, lr_scheduler, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(self.model_device_num), targets.to(
            self.model_device_num
        )
        self.optimizer.zero_grad()
        output = self.model(inputs)  # Forward pass
        loss = self.loss_criterion(output, targets)
        loss.backward()  # Backward pass
        self.optimizer.step()
        lr_scheduler.step()  # update the learning rate
        return loss

    def _lr_exp_stepper(self, x):
        """Exponentially increase learning rate as part of strategy to find the
        optimum.
        Taken from
        https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
        """
        return math.exp(
            x * self.log_lr_ratio / (self.lr_find_epochs * len(self.training_loader))
        )

    def _lr_finder(self, lr_scheduler, smoothing=0.05):
        lr_find_loss = []
        lr_find_lr = []
        iters = 0

        self.model.train()
        logging.info(
            f"Training for {self.lr_find_epochs} epochs to create a learning "
            "rate plot."
        )
        for i in range(self.lr_find_epochs):
            for batch in tqdm(
                self.training_loader,
                desc=f"Epoch {i + 1}, batch number",
                bar_format=cfg.TQDM_BAR_FORMAT,
            ):
                loss = self._train_one_batch(lr_scheduler, batch)
                lr_step = self.optimizer.state_dict()["param_groups"][0]["lr"]
                lr_find_lr.append(lr_step)
                if iters == 0:
                    lr_find_loss.append(loss)
                else:
                    loss = smoothing * loss + (1 - smoothing) * lr_find_loss[-1]
                    lr_find_loss.append(loss)
                if loss > 1 and iters > len(self.training_loader) // 1.333:
                    break
                iters += 1

        return lr_find_loss, lr_find_lr

    @staticmethod
    def _find_lr_from_graph(
        lr_find_loss: torch.Tensor, lr_find_lr: torch.Tensor
    ) -> float:
        """Calculates learning rate corresponsing to minimum gradient in graph
        of loss vs learning rate.
        Args:
            lr_find_loss (torch.Tensor): Loss values accumulated during training
            lr_find_lr (torch.Tensor): Learning rate used for mini-batch
        Returns:
            float: The learning rate at the point when loss was falling most steeply
            divided by a fudge factor.
        """
        default_min_lr = cfg.DEFAULT_MIN_LR  # Add as default value to fix bug
        # Get loss values and their corresponding gradients, and get lr values
        for i in range(0, len(lr_find_loss)):
            if lr_find_loss[i].is_cuda:
                lr_find_loss[i] = lr_find_loss[i].cpu()
            lr_find_loss[i] = lr_find_loss[i].detach().numpy()
        losses = np.array(lr_find_loss)
        # Smooth the losses
        kernel_size = 10
        kernel = np.ones(kernel_size) / kernel_size
        losses_convolved = np.convolve(losses, kernel, mode='same')
        try:
            gradients = np.gradient(losses_convolved)
            min_gradient = gradients.min()
            if min_gradient < 0:
                min_loss_grad_idx = gradients.argmin()
            else:
                logging.info(
                    f"Minimum gradient: {min_gradient} was positive, returning default value instead."
                )
                return default_min_lr
        except Exception as e:
            logging.info(f"Failed to compute gradients, returning default value. {e}")
            return default_min_lr
        min_lr = lr_find_lr[min_loss_grad_idx]
        return min_lr / cfg.LR_DIVISOR

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
            self.avg_eval_scores,
        )
        with open(output_dir / f"{model_out_path.stem}_train_stats.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(("Epoch", "Train Loss", "Valid Loss", "Eval Score"))
            for row in rows:
                writer.writerow(row)
