import logging

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', model_dict={}, best_score=None, use_accuracy=False, distributed=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.val_loss_min = np.inf if best_score is None else best_score * -1
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.use_accuracy = use_accuracy
        self.distributed = distributed
        self.model_struc_dict = model_dict # Dictionary with parameters controlling architecture

    def __call__(self, val_loss, model, optimizer, class_names, eval_score):

        if not self.use_accuracy:
            score = -val_loss
        else:
            score = eval_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, class_names)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.use_accuracy:
                logging.info(
                f'Accuracy Increased ({self.best_score:.6f} --> {score:.6f}).  Saving model to {self.path}')
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, class_names)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, class_names):
        '''Saves model when validation loss decrease.'''
        model_state_dict = None
        if self.verbose and not self.use_accuracy:
            logging.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {self.path}')
        if self.distributed:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        model_dict = {
            "model_state_dict": model_state_dict,
            "model_struc_dict": self.model_struc_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_val": val_loss,
            "class_names": class_names,
        }        
        torch.save(model_dict, self.path)
        self.val_loss_min = val_loss
        