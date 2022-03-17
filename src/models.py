import torch
from pytorch_lightning import LightningModule
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from torch.optim import Adam, lr_scheduler


class InverseSquareRootSchedule:
    def __init__(self, lr: float, warmup_updates: int):
        self.end_lr = lr
        self.warmup_updates = warmup_updates

        self.lr_step = 1 / self.warmup_updates
        self.decay_factor = self.warmup_updates**0.5

    def get_ratio(self, step: int):
        if step < self.warmup_updates:
            return self.lr_step * step
        else:
            return self.decay_factor * step**-0.5


class S2TLightningModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        # self.model = nn.Linear(28353980, 768)
        self.lr = hparams["module"]["optim"]["lr"]
        self.lr_func = InverseSquareRootSchedule(
            lr=self.lr, warmup_updates=hparams["module"]["optim"]["warmup_updates"]
        )

    def forward(self, input_values, labels):
        """
        Prediction step. The arguments are the contents of the batch.
        """
        # import pdb; pdb.set_trace()
        return self.model(input_values, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        batch is a dictionary.
        batch_idx is automatically assigned.
        Return loss or a dictionary like {"loss": xx.x, "metric1": yy.y, ...}.
        """
        model_output = self(**batch)
        # NOTE: self calls forward

        loss = model_output.loss
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self(**batch)

        loss = model_output.loss
        self.log("valid/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        input_values = self(**batch)

        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = self.processor.decode(predicted_ids[0])
        print(transcription)
        # TODO
        # save_transcription()

        loss = self.model(input_values, labels=batch["labels"]).loss
        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.lr,
        )

        scheduler = {
            "scheduler": lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self.lr_func.get_ratio
            ),
            "interval": "step",
            "name": "lr_scheduler",
        }

        return [optimizer], [scheduler]
