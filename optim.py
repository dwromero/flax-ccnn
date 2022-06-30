from omegaconf import OmegaConf
import optax


def construct_optimizer(
    cfg: OmegaConf,
):
    # Construct lr_scheduler
    lr_scheduler = construct_lr_scheduler(cfg)

    optimizer_type = cfg.optimizer.name.lower()
    optimizer_type = getattr(optax, optimizer_type)
    optimizer = optimizer_type(
        learning_rate=lr_scheduler,
        weight_decay=0.0,
    )
    # Weight decay is implemented somewhere else.
    return optimizer, lr_scheduler


def construct_lr_scheduler(
    cfg: OmegaConf,
):
    warmup_iterations = (
            cfg.scheduler.warmup_epochs * cfg.scheduler.iters_per_train_epoch
    )
    total_iterations = cfg.scheduler.total_train_iters

    if cfg.scheduler.name == 'cosine':
        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.optimizer.lr,
            warmup_steps=warmup_iterations,
            decay_steps=total_iterations - warmup_iterations,
            end_value=0.0,
        )
    else:
        print('WARNING: No scheduler will be used.')
        lr_scheduler = optax.constant_schedule(value=cfg.optimizer.lr)
    return lr_scheduler









