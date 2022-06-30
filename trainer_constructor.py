import os.path
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state, checkpoints
import hydra

from models.wrappers import TrainState

from tqdm.auto import tqdm

# typing
from omegaconf import OmegaConf
from typing import Any

import wandb


class Trainer:
    def __init__(
        self,
    ):
        super().__init__()
        self.global_train_iteration = 0
        self.log_interval = 50
        self.ckpt_dir = None
        self.max_epochs_no_improvement = 100

    def step_logger(self, model, loss, accuracy, epoch):
        wandb.log(
            {
                'train/loss_step': loss,
                'train/acc_step': accuracy,
                'lr': model._lr_scheduler(self.global_train_iteration),
                'epoch': epoch,
            },
            step=self.global_train_iteration,
        )

    def epoch_logger(self, model, loss, accuracy, epoch, mode):
        wandb.log(
            {
                f'{mode}/loss_epoch': loss,
                f'{mode}/acc_epoch': accuracy,
            },
            step=self.global_train_iteration,
        )

    def on_train_start(self, model):
        model.on_train_start()

    def on_train_end(self, model):
        model.on_train_end()
        pass # TODO: Clean up local and cloud artifacts

    def train(self, model, datamodule, no_epochs):
        # Next, create instances of the dataloaders
        train_dataloader = datamodule.train_dataloader
        val_dataloader = datamodule.val_dataloader
        # call on_train_start()
        self.on_train_start(model)
        # Track best val accuracy:
        best_val_accuracy = 0.0
        epochs_no_improvement = 0
        for epoch in tqdm(range(1, no_epochs + 1)):
            self.train_epoch(model, train_dataloader, epoch)
            val_acccuracy = self.validate(model, val_dataloader, epoch)
            if val_acccuracy >= best_val_accuracy:
                best_val_accuracy = val_acccuracy
                print(f'Best validation acc: {best_val_accuracy}')
                self.save_model(model, aliases=['best', 'latest'], value=best_val_accuracy, epoch=epoch)
                # Restart counter epochs_no_improvement
                epochs_no_improvement = 0
            else:
                self.save_model(model, aliases=['latest'], value=val_acccuracy, epoch=epoch)
                epochs_no_improvement += 1
            # Early stopping
            if epochs_no_improvement == self.max_epochs_no_improvement:
                print('Stopping training. Epochs without improvement matched the max epochs without '
                      f'improvement of {self.max_epochs_no_improvement}.')
                break
            # Return best results
            print(f'Best Validation Accuracy: {best_val_accuracy}')
        self.on_train_end(model)

    def train_epoch(self, model, train_dataloader, epoch):
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(list)
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch} / Training', leave=False):
            model.state, loss, accuracy = model.training_step(model.state, batch)
            metrics['loss'].append(loss)
            metrics['acc'].append(accuracy)
            # Log step metrics
            if self.global_train_iteration % self.log_interval == 0:
                self.step_logger(model, loss, accuracy, epoch)
            # Increase global iteration counter
            self.global_train_iteration += 1
        avg_metrics = defaultdict(float)
        for key in metrics:
            avg_metrics[key] = np.stack(jax.device_get(metrics[key])).mean()
        # Log epoch metrics
        self.epoch_logger(model, avg_metrics["loss"], avg_metrics["acc"], epoch, mode='train')
        print(f'Epoch {epoch} / Train / Loss = {avg_metrics["loss"]:.2f}, Accuracy = {avg_metrics["acc"]:.4f}')

    def validate(self, model, val_dataloader, epoch):
        # Validate model on the validation set and log avg loss and accuracy
        metrics = defaultdict(list)
        for batch in tqdm(val_dataloader, desc=f'Epoch {epoch} / Training', leave=False):
            _, loss, accuracy = model.validation_step(model.state, batch)
            metrics['loss'].append(loss)
            metrics['acc'].append(accuracy)
        avg_metrics = defaultdict(float)
        for key in metrics:
            avg_metrics[key] = np.stack(jax.device_get(metrics[key])).mean()
        # Log epoch metrics
        self.epoch_logger(model, avg_metrics["loss"], avg_metrics["acc"], epoch, mode='val')
        print(f'Epoch {epoch} / Val / Loss = {avg_metrics["loss"]:.2f}, Accuracy = {avg_metrics["acc"]:.4f}')
        return avg_metrics['acc']

    def test(self, model, test_dataloader):
        # Validate model on the validation set and log avg loss and accuracy
        metrics = defaultdict(list)
        for batch in tqdm(test_dataloader, desc=f'Test', leave=False):
            _, loss, accuracy = model.test_step(model.state, batch)
            metrics['loss'].append(loss)
            metrics['acc'].append(accuracy)
        avg_metrics = defaultdict(float)
        for key in metrics:
            avg_metrics[key] = np.stack(jax.device_get(metrics[key])).mean()
        # Log epoch metrics
        self.epoch_logger(model, avg_metrics["loss"], avg_metrics["acc"], epoch=None, mode='test')
        print(f'Test / Loss = {avg_metrics["loss"]:.2f}, Accuracy = {avg_metrics["acc"]:.4f}')
        return avg_metrics['acc']

    def save_model(self, model, aliases, value, epoch):
        model.save(aliases, value, epoch)

    def load_model(self, model, alias):
        model.load(alias)


def construct_trainer(
        cfg: OmegaConf,
) -> Trainer:
    return Trainer()

