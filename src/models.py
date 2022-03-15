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
        self.lr = hparams["module"]["optim"]["lr"]
        self.lr_func = InverseSquareRootSchedule(
            lr=self.lr, warmup_updates=hparams["module"]["optim"]["warmup_updates"]
        )

    def forward(self, waveform, sr, src_utt, tgt_utt, spk_id, tgt_lang, utt_id):
        """
        Defines prediction step. arguments are the contents of batch
        """
        input_values = self.processor(waveform, sampling_rate=sr, return_tensors="pt").input_values
        return input_values

    def training_step(self, batch, batch_idx):
        """
        batch is a dictionary
        batch_idx is automatically assigned
        returns loss or a dictionary like {"loss": xx.x, "metric1": yy.y, ...}
        """
        input_values = self(**batch)  # this calls forward

        with self.processor.as_target_processor():
            # target is src_utt because the task is ASR
            labels = self.processor(batch["src_utt"], return_tensors="pt").input_ids

        loss = self.model(input_values, labels=labels).loss
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_values = self(**batch)

        with self.processor.as_target_processor():
            labels = self.processor(batch["src_utt"], return_tensors="pt").input_ids

        loss = self.model(input_values, labels=labels).loss
        self.log("valid/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        input_values = self(**batch)

        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # TODO: polish this
        # transcription = self.processor.decode(predicted_ids[0])
        # save_transcription()

        with self.processor.as_target_processor():
            labels = self.processor(batch["src_utt"], return_tensors="pt").input_ids

        loss = self.model(input_values, labels=labels).loss
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
