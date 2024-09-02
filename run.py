import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import torch.distributed as dist
import torch
import tensorboard

from .parser import load_config, parse_args
from .utils import logging
from .tasks import load_task
from .datasets.get_datamodule import get_dm
import sys
sys.path.append("..")

logger = logging.get_logger(__name__)


def cleanup():
    dist.destroy_process_group()


def train(args, cfg):
    seed_everything(cfg.seed)

    # setup logger
    log_path = '{}'.format(args.exp_name)

    # dataset module
    dm = get_dm(cfg)
    dm.setup('fit')
    dm.train_dataloader()  # initialize dm.train_loader
    steps_in_epoch = len(dm.train_loader) // cfg.num_gpus
    print('steps_in_epoch: ', steps_in_epoch)

    # task module
    task = load_task(cfg, steps_in_epoch)

    # trainer setting
    tb_logger = TensorBoardLogger(save_dir=cfg.outdir, name='lightning_logs', version=log_path)
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.train.checkpoint_metric,
        mode=cfg.train.checkpoint_mode,
        save_last=True,
        save_top_k=1
    )

    learning_rate_callback = LearningRateMonitor()
    trainer_args = {}


    strategy = DDPStrategy(find_unused_parameters=False)

    if cfg.train.strategy == 'ddp':
        trainer_args = {
            'accelerator': 'gpu',
            'devices': cfg.num_gpus,   # devices=[1, 3] to specify GPUs
            'strategy': strategy
        }
    elif cfg.train.strategy == 'cpu':
        trainer_args = {}
    
    #gpu stats
    gpu_stats = DeviceStatsMonitor() 
    trainer = Trainer(
        max_epochs=cfg.solver.num_epochs,
        benchmark=True,
        fast_dev_run=cfg.fast_dev_run,
        limit_train_batches=cfg.train.limit_train_batches,  # to avoid tensorboard issue
        limit_val_batches=cfg.train.limit_val_batches,  # to avoid tensorboard issue
        val_check_interval=cfg.train.val_check_interval,
        enable_progress_bar=cfg.enable_progress_bar,
        logger=tb_logger,
        #accumulate_grad_batches=gradient_accumulation_steps,
        callbacks=[learning_rate_callback, checkpoint_callback, gpu_stats],
        **trainer_args,
    )

    trainer.fit(model=task, datamodule=dm, ckpt_path=cfg.ckpt_path)
    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    cleanup()
    return checkpoint_callback.best_model_path


def test(args, cfg, ckpt_path):
    seed_everything(cfg.seed)

    # setup logger
    log_path = '{}/test'.format(args.exp_name)

    # dataset module
    dm = get_dm(cfg)

    # task module
    task = load_task(cfg)

    tb_logger = TensorBoardLogger(save_dir=cfg.outdir, name='lightning_logs', version=log_path)

    # devices=1 to avoid distributed sampler.
    trainer = Trainer(
        accelerator = 'gpu',
        logger = tb_logger,
        devices = 1,
        limit_test_batches = cfg.test.limit_test_batches,
    )
    trainer.test(model=task, datamodule=dm, ckpt_path=ckpt_path)


def val(args, cfg, ckpt_path):
    seed_everything(cfg.seed)

    # setup logger
    log_path = '{}/val'.format(args.exp_name)

    # dataset module
    dm = get_dm(cfg)
    dm.setup('fit')

    # task module
    task = load_task(cfg)

    tb_logger = TensorBoardLogger(save_dir=cfg.outdir, name='lightning_logs', version=log_path)

    trainer_args = {}
    if cfg.train.strategy == 'ddp':
        trainer_args = {
            'accelerator': 'gpu',
            'devices': cfg.num_gpus,   # devices=[1, 3] to specify GPUs
            'strategy': 'ddp',
            # 'strategy': DDPStrategy(find_unused_parameters=False, static_graph=True, gradient_as_bucket_view=True),
        }
    elif cfg.train.strategy == 'cpu':
        trainer_args = {}

    trainer = Trainer(
        limit_val_batches=cfg.train.limit_val_batches,  # to avoid tensorboard issue
        logger=tb_logger,
        **trainer_args,
    )

    trainer.validate(model=task, dataloaders=dm.val_dataloader(), ckpt_path=ckpt_path)


def main():
    # parse arg and cfg
    args = parse_args()
    cfg = load_config(args)

    # set seed
    seed_everything(cfg.seed)

    ckpt_path = cfg.ckpt_path
    if cfg.val.val_only:
        val(args, cfg, ckpt_path)
    else:
        if cfg.train.enable:
            ckpt_path = train(args, cfg)
        if cfg.test.enable:
            test(args, cfg, ckpt_path)


if __name__ == "__main__":
    main()