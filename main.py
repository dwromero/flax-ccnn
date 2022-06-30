# torch
import jax

import os

# Project
import wandb

import ckconv
from dataset_constructor import construct_datamodule
from model_constructor import construct_model
from trainer_constructor import construct_trainer

# Loggers
# from pytorch_lightning.loggers import WandbLogger

# Configs
import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(
    cfg: OmegaConf,
):
    # We possibly want to add fields to the config file. Thus, we set struct to False.
    OmegaConf.set_struct(cfg, False)

    # Set seed # TODO
    main_rng = jax.random.PRNGKey(seed=cfg.seed)

    # Check number of available devices
    cfg.train.avail_gpus = jax.device_count()

    # Construct data modules
    datamodule = construct_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    # Append no of iteration to the cfg file for the definition of the schedulers
    distrib_batch_size = cfg.train.batch_size
    if cfg.train.distributed:
        distrib_batch_size *= cfg.train.avail_gpus
    cfg.scheduler.iters_per_train_epoch = (
            len(datamodule.train_dataset) // distrib_batch_size
    )
    cfg.scheduler.total_train_iters = (
            cfg.scheduler.iters_per_train_epoch * cfg.train.epochs
    )

    # Construct model
    model = construct_model(cfg, datamodule)
    # initialize model with datamodules & pack the parameters in a TrainState
    model.create_state(datamodule.train_dataloader)
    # model.create_state()

    # Create trainer
    trainer = construct_trainer(cfg)

    # Initialize wandb
    if not cfg.train.do or cfg.debug:
        mode = 'offline'
    else:
        mode = 'online'
    wandb.init(
        project=cfg.wandb.project,
        config=ckconv.utils.flatten_configdict(cfg),
        entity=cfg.wandb.entity,
        save_code=True,
        mode=mode,
    )

    # Train
    if cfg.train.do:
        trainer.train(model, datamodule, cfg.train.epochs)
        # Load best model
        trainer.load_model(model, alias='best')

    # Validate and test before finishing
    trainer.validate(model, datamodule.val_dataloader, epoch=0)
    trainer.test(model, datamodule.test_dataloader)







if __name__ == "__main__":
    main()
