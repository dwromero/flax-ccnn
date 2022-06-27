import jax
import optax
from flax import linen as nn
from flax.training import train_state

import models
from optim import construct_optimizer

# typing
from omegaconf import OmegaConf
from typing import Any


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any


class WrapperBase:

    def __init__(
        self,
        cfg: OmegaConf,
    ):
        super().__init__()

        # Configuration file
        self.cfg = cfg

        # Network parameters
        self.init_params = None
        self.init_batch_stats = None
        self.state = None

    def initialize_network(
            self,
            train_dataloader,
    ):
        init_rng = jax.random.PRNGKey(self.cfg.seed)
        example_data, _ = next(iter(train_dataloader))
        variables = self.network.init(init_rng, example_data, train=True)
        self.init_params, self.init_batch_stats = variables['params'], variables['batch_stats']

    def create_state(self):
        self.state = TrainState.create(
            apply_fn=self.network.apply,
            params=self.init_params,
            batch_stats=self.init_batch_stats,
            tx=self.configure_optimizers(),
        )

    def configure_optimizers(self):
        return construct_optimizer(self.cfg)

    def __call__(self, x):
        raise NotImplementedError

    def on_train_start(self):
        raise NotImplementedError

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
        loss_fn = lambda params: self.step(params, state.batch_stats, batch, train=True)
        # Get loss, gradients for loss, and other outputs of loss function
        (loss, (accuracy, new_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        # Update parameters and batch statistics
        state = self.state.apply_gradients(grads=grads, batch_stats=new_state['batch_stats'])
        return state, loss, accuracy

    def validation_step(self, state, batch):
        # Return the accuracy for a single batch
        loss, (accuracy, _) = self.step(self.state.params, self.state.batch_stats, batch, train=False)
        return None, loss, accuracy

    # Eval function
    def test_step(self, state, batch):
        return self.validation_step(state, batch)








