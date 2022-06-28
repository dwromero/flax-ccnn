from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state, checkpoints

from tqdm.auto import tqdm

# typing
from omegaconf import OmegaConf
from typing import Any


class Trainer:
    def __init__(
        self,
    ):
        super().__init__()

    def train(self, model, datamodule, no_epochs):
        # Next, create instances of the dataloaders
        train_dataloader = datamodule.train_dataloader
        val_dataloader = datamodule.val_dataloader
        # Track best val accuracy:
        best_val_accuracy = 0.0
        for epoch in tqdm(range(1, no_epochs + 1)):
            self.train_epoch(model, train_dataloader, epoch)
            val_acccuracy = self.val_epoch(model, val_dataloader, epoch)
            if val_acccuracy >= best_val_accuracy:
                best_val_accuracy = val_acccuracy
                print(f'Best validation acc: {best_val_accuracy}')
                # self.save_model(step=epoch)
            # self.logger.flush()

    @staticmethod
    def train_epoch(model, train_dataloader, epoch):
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(list)
        for batch in tqdm(train_dataloader, desc='Training', leave=False):
            model.state, loss, accuracy = model.training_step(model.state, batch)
            metrics['loss'].append(loss)
            metrics['acc'].append(accuracy)
        avg_metrics = defaultdict(float)
        for key in metrics:
            avg_metrics[key] = np.stack(jax.device_get(metrics[key])).mean()
        print(f'Epoch {epoch} / Train / Loss = {avg_metrics["loss"]:.2f}, Accuracy = {avg_metrics["acc"]:.4f}')
            # self.logger.add_scalar('train/' + key, avg_train_metric, global_step=epoch)

    @staticmethod
    def val_epoch(model, val_dataloader, epoch):
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(list)
        for batch in tqdm(val_dataloader, desc='Validation', leave=False):
            _, loss, accuracy = model.validation_step(model.state, batch)
            metrics['loss'].append(loss)
            metrics['acc'].append(accuracy)
        avg_metrics = defaultdict(float)
        for key in metrics:
            avg_metrics[key] = np.stack(jax.device_get(metrics[key])).mean()
            # self.logger.add_scalar('val/' + key, avg_val_metric, global_step=epoch)
        print(f'Epoch {epoch} / Val / Loss = {avg_metrics["loss"]:.2f}, Accuracy = {avg_metrics["acc"]:.4f}')
        return avg_metrics['acc']

    def save_model(self, step=0):
        # Save current model at certain iterations #TODO: Wandb integration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.state.params,
                                            'batch_stats': self.state.batch_stats},
                                    step=step,
                                    overwrite=True)


def construct_trainer(
        cfg: OmegaConf,
) -> Trainer:
    return Trainer()

