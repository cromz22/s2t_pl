from pytorch_lightning import LightningDataModule
from typing import Optional
from torch.utils.data import DataLoader
from torch import LongTensor

from mustcv2_dataset import MuSTCv2Dataset


def collate_fn(batch):
    """
    change dataset items to dict of tensors
    TODO: need change
    """
    collated_batch = {}
    for sample in batch:
        for key, value in sample.items():
            collated_batch.setdefault(key, []).append(value)
    return {key: LongTensor(value) for key, value in collated_batch.items()}


class MuSTCDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()

        self.data_dir = hparams["data"]["data_dir"]
        self.tgt_lang = hparams["data"]["tgt_lang"]

        self.num_workers = 4

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

    def train_dataloader(self):
        """
        turn dataset into batch
        """
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
