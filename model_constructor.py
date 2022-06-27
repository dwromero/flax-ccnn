
import models
import models.wrappers as wrappers

# typing
from omegaconf import OmegaConf
from datamodules import DataModule


def construct_model(
    cfg: OmegaConf,
    datamodule: DataModule,
):
    """
    :param cfg: configuration file
    :return: An instance of torch.nn.Module
    """
    # Get parameters of model from task type
    data_dim = datamodule.data_dim
    in_channels = datamodule.input_channels
    out_channels = datamodule.output_channels
    data_type = datamodule.data_type

    # Get type of model from task type
    net_type = f"{cfg.net.type}_{data_type}"

    # Overwrite data_dim in cfg.net
    cfg.net.data_dim = data_dim

    # Print automatically derived model parameters.
    print(
        f"Automatic Parameters:\n dataset = {cfg.dataset.name}, "
        f"net_name = {net_type},"
        f" data_dim = {data_dim}"
        f" in_channels = {in_channels},"
        f" out_chanels = {out_channels}."
    )
    if out_channels == 2:
        print(
            "The model will output one single channel. We use BCEWithLogitsLoss for training."
        )
        out_channels = 1

    # Construct model with the corresponding task wrapper
    model = wrappers.ClassificationWrapper(
        cfg=cfg,
        data_dim=data_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        data_type=data_type,
    )
    # return model
    return model
