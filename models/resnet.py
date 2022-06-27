import jax.numpy as jnp
import flax.linen as nn
from . import modules
from functools import partial

# project
import ckconv

# typing
from omegaconf import OmegaConf


class ResNetBase(nn.Module):
    in_channels: int
    out_channels: int
    net_cfg: OmegaConf
    kernel_cfg: OmegaConf
    conv_cfg: OmegaConf
    mask_cfg: OmegaConf

    def setup(self):
        # Unpack arguments from net_config
        hidden_channels = self.net_cfg.no_hidden
        no_blocks = self.net_cfg.no_blocks
        data_dim = self.net_cfg.data_dim
        norm = self.net_cfg.norm
        dropout = self.net_cfg.dropout
        dropout_in = self.net_cfg.dropout_in
        block_type = self.net_cfg.block.type
        block_prenorm = self.net_cfg.block.prenorm
        block_width_factors = self.net_cfg.block_width_factors
        nonlinearity = self.net_cfg.nonlinearity

        # Define dropout_in
        self.dropout_in = nn.Dropout(rate=dropout_in, deterministic=False)

        # Unpack conv_type
        conv_type = self.conv_cfg.type
        # Define partials for types of convs
        ConvType = partial(
            getattr(nn, conv_type),
            kernel_size=(5,) * data_dim,
        )
        # -------------------------

        # Define NormType
        NormType = getattr(nn, norm)

        # Define NonlinearType
        NonlinearType = getattr(nn, nonlinearity)

        # Create Input Layers
        self.conv1 = ConvType(features=hidden_channels)
        self.norm1 = NormType()
        self.nonlinear = NonlinearType # TODO

        # Create Blocks
        # -------------------------
        if block_type == "default":
            BlockType = modules.ResNetBlock
        else:
            BlockType = getattr(modules, f"{block_type}Block")
        # 1. Create vector of width_factors:
        # If value is zero, then all values are one
        if block_width_factors[0] == 0.0:
            width_factors = (1,) * no_blocks
        else:
            width_factors = [
                (factor,) * n_blcks
                for factor, n_blcks in ckconv.utils.pairwise_iterable(block_width_factors)
            ]
            width_factors = [
                factor for factor_tuple in width_factors for factor in factor_tuple
            ]
        if len(width_factors) != no_blocks:
            raise ValueError(
                "The size of the width_factors does not matched the number of blocks in the network."
            )
        # 2. Create blocks
        blocks = []
        for i in range(no_blocks):
            print(f"Block {i}/{no_blocks}")

            if i == 0:
                input_ch = hidden_channels
                hidden_ch = int(hidden_channels * width_factors[i])
            else:
                input_ch = int(hidden_channels * width_factors[i - 1])
                hidden_ch = int(hidden_channels * width_factors[i])

            blocks.append(
                BlockType(
                    in_channels=input_ch,
                    out_channels=hidden_ch,
                    ConvType=ConvType,
                    NonlinearType=NonlinearType,
                    NormType=NormType,
                    dropout=dropout,
                    prenorm=block_prenorm,
                )
            )

        self.blocks = blocks
        # -------------------------

        # Define Output Layers:
        self.out_layer = nn.Dense(
            features=self.out_channels,
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=nn.initializers.zeros,
        )
        # -------------------------
        if block_type == "S4" and block_prenorm:
            self.out_norm = NormType()
        else:
            self.out_norm = lambda x: x

        # Save variables in self
        self.data_dim = data_dim

    def __call__(self, x, train):
        raise NotImplementedError


class ResNet_sequence(ResNetBase):
    def __call__(self, x, train):
        # Dropout in
        # x = self.dropout_in(x) TODO
        # First layers
        out = self.nonlinear(
            self.norm1(self.conv1(x), use_running_average=not train),
        )
        # Blocks
        for block in self.blocks:
            out = block(out, train)
        # Final layer on last sequence element
        out = self.out_norm(out, use_running_average=not train)
        # Take the mean of all predictions until the last element
        out = jnp.mean(out, axis=1)
        # Pass through final projection layer, squeeze & return
        out = self.out_layer(out)
        return out


class ResNet_image(ResNetBase):
    def __call__(self, x, train):
        # Dropout in
        # x = self.dropout_in(x) TODO
        # First layers
        out = self.nonlinear(
            self.norm1(self.conv1(x), use_running_average=not train),
        )
        # Blocks
        for block in self.blocks:
            out = block(out, train)
        # Final layer on last sequence element
        out = self.out_norm(out, use_running_average=not train)
        # Pool
        out = jnp.mean(out, axis=(-3, -2))
        # Final layer
        out = self.out_layer(out)
        return out.squeeze()



















