import torch
from pytorch_lightning import LightningDataModule
from typing import Optional
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

from mustcv2_dataset import MuSTCv2Dataset


class MuSTCDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()

        self.data_dir = hparams["data"]["data_dir"]
        self.tgt_lang = hparams["data"]["tgt_lang"]

        self.num_workers = 0
        self.batch_size = 4

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MuSTCv2Dataset(
                root=self.data_dir, tgt_lang=self.tgt_lang, split="train"
            )
            self.val_dataset = MuSTCv2Dataset(
                root=self.data_dir, tgt_lang=self.tgt_lang, split="dev"
            )

        if stage == "test" or stage is None:
            self.test_dataset = MuSTCv2Dataset(
                root=self.data_dir, tgt_lang=self.tgt_lang, split="tst-COMMON"
            )

    def collate_fn(self, batch_list):
        """
        Turn a list of dict of lists or tensors into one big dict of tensors. (default_collate)
        Each sample is dict(waveform=waveform, sr=sr, labels=labels), as returned by MuSTCv2Dataset.__getitem__.
        Perform padding if necessary.
        """
        waveforms = []
        labels = []
        for sample in batch_list:
            waveforms.append(sample["waveform"])
            labels.append(sample["labels"])

        sr = batch_list[0]["sr"]
        # Assume all waveforms in the batch have the same sampling rate.

        # import pdb; pdb.set_trace()
        input_values = self.processor(
            waveforms, sampling_rate=sr, return_tensors="pt", padding=True
        ).input_values
        # NOTE: waveform can be multiple waveforms. In that case, padding is performed to fit the longest waveform.

        return dict(
            input_values=input_values,
            # labels=torch.stack(labels)
            labels=labels,
        )

    def train_dataloader(self):
        """
        turn dataset into batch
        """
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
