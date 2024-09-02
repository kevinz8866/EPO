import torch
from ..optimizers import lr_scheduler
from ..models.language_model import LlamaText, LlamaHead
from pytorch_lightning.core import LightningModule


class VideoTask(LightningModule):
    def __init__(self, cfg, steps_in_epoch):
        super().__init__()
        self.cfg = cfg
        self.steps_in_epoch = steps_in_epoch
        self.save_hyperparameters()
        self.model = self.build_model()

    def build_model(self):
        if self.cfg.model.model == 'language':
            model = LlamaText(self.cfg)
            return model
        elif self.cfg.model.model == 'language_head':
            model = LlamaHead(self.cfg)
            return model
        else:
            raise NotImplementedError(f'model {self.cfg.model.model} not implemented')

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def forward(self, inputs):
        return self.model(inputs)

    def configure_optimizers(self):
        return lr_scheduler.lr_factory(self.model, self.cfg, self.steps_in_epoch)
