import jax
import optax
import flax
import wandb
from flax import linen as nn
import jax.numpy as jnp
from flax.training import train_state, checkpoints

import glob
import hydra
import os

import ckconv
import models
from optim import construct_optimizer

# typing
from omegaconf import OmegaConf
from typing import Any


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: flax.core.FrozenDict[str, Any]


class WrapperBase:

    def __init__(
        self,
        cfg: OmegaConf,
    ):
        super().__init__()
        # Configuration file
        self._lr_scheduler = None
        self.cfg = cfg
        self.state = None
        self.no_params = None

    def initialize_network(
            self,
            train_dataloader,
    ):
        init_rng = jax.random.PRNGKey(self.cfg.seed)
        example_data, _ = next(iter(train_dataloader))
        init_variables = self.network.init(init_rng, example_data, train=True)
        return init_variables

    def create_state(self, train_dataloader):
        # Initialize network
        init_variables = self.initialize_network(train_dataloader)
        optimizer, lr_scheduler = construct_optimizer(self.cfg)
        # Use initialization parameters to create state
        self.state = TrainState.create(
            apply_fn=self.network.apply,
            params=init_variables['params'],
            batch_stats=init_variables['batch_stats'],
            tx=optimizer,
        )
        self._lr_scheduler = lr_scheduler # Used for logging purposes

    def on_train_start(self):
        # Log number of parameters
        no_params = ckconv.utils.no_params(self.state)
        wandb.summary['no_params'] = no_params
        self.no_params = no_params

        # Log code
        code = wandb.Artifact(
            f"source-code-{wandb.run.name}", type="code"
        )
        # Get paths
        paths = glob.glob(
            hydra.utils.get_original_cwd() + "/**/*.py",
            recursive=True,
        )
        paths += glob.glob(
            hydra.utils.get_original_cwd() + "/**/*.yaml",
            recursive=True,
        )
        # Filter paths
        paths = list(filter(lambda x: "outputs" not in x, paths))
        paths = list(filter(lambda x: "wandb" not in x, paths))
        # Get all source files
        for path in paths:
            code.add_file(
                path,
                name=path.replace(f"{hydra.utils.get_original_cwd()}/", ""),
            )
        # Use the artifact
        if not wandb.run.offline:
            wandb.run.use_artifact(code)

    def save(self, aliases, value, epoch):
        ckpt_dir = os.path.join(wandb.run.dir, f'model-{wandb.run.name}')
        # Save current model with the provided aliases
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir,
                                    target={'params': self.state.params,
                                            'batch_stats': self.state.batch_stats},
                                    step=epoch,
                                    overwrite=True)
        # Log to wandb
        metadata = {
            'score': value,
        }
        checkpoint = wandb.Artifact(
            f"model-{wandb.run.name}", type="model", metadata=metadata,
        )
        checkpoint.add_file(os.path.join(ckpt_dir, f'checkpoint_{epoch}'))
        wandb.log_artifact(checkpoint, aliases=aliases)

    def load(self, alias):
        # Download artifact with given alias model
        artifact = f"{wandb.run.entity}/{wandb.run.project}/model-{wandb.run.name}:{alias}"
        artifact = wandb.Api().artifact(artifact)
        ckpt_path = artifact.download(
            root=hydra.utils.get_original_cwd() + f"/artifacts/{artifact.name}"
        )
        # Restore checkpoint
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)
        self.state = TrainState.create(
            apply_fn=self.network.apply,
            params=state_dict['params'],
            batch_stats=state_dict['batch_stats'],
            tx=self.state.tx,
        )

    def on_train_end(self):
        pass

    def training_step(self, state, batch):
        raise NotImplementedError

    def validation_step(self, state, batch):
        raise NotImplementedError

    def test_step(self, state, batch):
        raise NotImplementedError

    def training_epoch_end(self, train_step_outputs):
        raise NotImplementedError

    def validation_epoch_end(self, validation_step_outputs):
        raise NotImplementedError


class ClassificationWrapper(WrapperBase):
    def __init__(
        self,
        cfg: OmegaConf,
        data_dim: int,
        in_channels: int,
        out_channels: int,
        data_type: int,
    ):
        super().__init__(
            cfg=cfg,
        )

        # Construct network
        # Get type of model from task type
        net_type = f"{cfg.net.type}_{data_type}"
        NetworkClass = getattr(models, net_type)
        self.network = NetworkClass(
            in_channels=in_channels,
            out_channels=out_channels if out_channels != 2 else 1,
            net_cfg=cfg.net,
            kernel_cfg=cfg.kernel,
            conv_cfg=cfg.conv,
            mask_cfg=cfg.mask,
        )
        self.training_step = jax.jit(self.training_step)
        self.validation_step = jax.jit(self.validation_step)
        self.test_step = jax.jit(self.test_step)

    def step(self, params, batch_stats, batch, train):
        data, labels = batch
        labels_one_hot = jax.nn.one_hot(labels, num_classes=self.network.out_channels)
        # Run model. During training, we must update Norm statistics.
        outs = self.network.apply(
            {'params': params, 'batch_stats': batch_stats},
            data,
            train=train,
            mutable=['batch_stats'] if train else False,
        )
        logits, new_state = outs if train else (outs, None)
        loss = optax.softmax_cross_entropy(logits, labels_one_hot).mean()
        accuracy = (logits.argmax(axis=-1) == labels).mean()
        return loss, (accuracy, new_state)

    def training_step(self, state, batch):
        # Get loss, gradients for loss, and other outputs of loss function
        (loss, (accuracy, new_state)), grads = jax.value_and_grad(self.step, has_aux=True)\
            (state.params, state.batch_stats, batch, train=True)
        # Update parameters and batch statistics
        new_state = state.apply_gradients(grads=grads, batch_stats=new_state['batch_stats'])
        return new_state, loss, accuracy

    def validation_step(self, state, batch):
        # Return the accuracy for a single batch
        loss, (accuracy, _) = self.step(state.params, state.batch_stats, batch, train=False)
        return None, loss, accuracy

    def test_step(self, state, batch):
        return self.validation_step(state, batch)








