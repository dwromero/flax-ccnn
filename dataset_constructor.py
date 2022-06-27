# built-in
import os

# typing
from omegaconf import OmegaConf

# datamodules
import datamodules
from datamodules import DataModule


def construct_datamodule(
    cfg: OmegaConf,
) -> DataModule:

    # Define num_workers
    if cfg.no_workers == -1:
        cfg.no_workers = int(os.cpu_count() / 4)

    # Gather module from datamodules, create instance and return
    dataset_name = f"{cfg.dataset.name}DataModule"
    dataset = getattr(datamodules, dataset_name)
    datamodule = dataset(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        test_batch_size=cfg.test.batch_size_multiplier * cfg.train.batch_size,
        data_type=cfg.dataset.data_type,
        num_workers=cfg.no_workers,
        augment=cfg.dataset.augment,
        **cfg.dataset.params,
    )
    # Assert if the datamodule has the parameters needed for the model creation
    assert hasattr(datamodule, "data_dim")
    assert hasattr(datamodule, "input_channels")
    assert hasattr(datamodule, "output_channels")
    assert hasattr(datamodule, "data_type")
    return datamodule