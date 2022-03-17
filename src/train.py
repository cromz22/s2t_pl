import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from models import S2TLightningModule
from datamodules import MuSTCDataModule


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    seed_everything(seed=cfg.seed, workers=True)
    print(f"{cfg = }")

    model = S2TLightningModule(hparams=cfg)
    datamodule = MuSTCDataModule(hparams=cfg)

    wandb_logger = WandbLogger(**cfg.logger)

    callbacks = [
        EarlyStopping(**cfg.callbacks.early_stop),
        ModelCheckpoint(**cfg.callbacks.checkpoint),
        LearningRateMonitor(**cfg.callbacks.lr_monitor),
    ]

    trainer = Trainer(logger=wandb_logger, callbacks=callbacks, **cfg.trainer)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
