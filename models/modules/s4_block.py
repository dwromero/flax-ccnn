import flax.linen as nn


class S4Block(nn.Module):
    in_channels: int
    out_channels: int
    ConvType: nn.Module
    NonlinearType: nn.Module
    NormType: nn.Module
    dropout: float
    prenorm: bool = True

    def setup(self):
        # Conv layers
        self.conv = self.ConvType(features=self.out_channels)

        # Nonlinear layers
        self.nonlinears = [
                self.NonlinearType,
                self.NonlinearType,
            ]

        # Norm layers
        self.norm = self.NormType() # Channels derived from input

        # Linear layer
        self.linear = nn.Dense(
            features=self.out_channels,
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=nn.initializers.zeros,
        )

        # Dropout
        # self.dp = nn.Dropout(rate=self.dropout, deterministic=False) # TODO

        # Shortcut
        shortcut = []
        if self.in_channels != self.out_channels:
            shortcut.append(
                nn.Dense(
                    features=self.out_channels,
                    kernel_init=nn.initializers.kaiming_normal(),
                    bias_init=nn.initializers.zeros,
                )
            )
        else:
            shortcut.append(lambda x: x)
        self.shortcut = nn.Sequential(shortcut)

    def __call__(self, x, train):
        shortcut = self.shortcut(x)
        # if prenorm: Norm -> Conv -> Nonlinear -> Linear -> Sum
        # else: Conv -> Nonlinear -> Linear -> Sum -> Norm
        if self.prenorm:
            x = self.norm(x, use_running_average=not train)
        # x = self.nonlinears[1](
        #     self.linear(self.dp(self.nonlinears[0](self.conv(x))))
        # )
        x = self.nonlinears[1](
            self.linear(self.nonlinears[0](self.conv(x)))
        )
        x = x + shortcut
        if not self.prenorm:
            x = self.norm(x, use_running_average=not train)
        return x



















