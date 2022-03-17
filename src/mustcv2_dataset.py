import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
import soundfile as sf
from transformers import Wav2Vec2Processor


class MuSTCv2Dataset(Dataset):
    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    TGT_LANGUAGES = ["de", "zh", "ja"]

    def __init__(self, root: str, tgt_lang: str, split: str) -> None:
        assert split in self.SPLITS and tgt_lang in self.TGT_LANGUAGES
        _root = Path(root) / f"en-{tgt_lang}" / "data" / split
        assert _root.is_dir()
        wav_root = _root / "wav" / "divided"
        txt_root = _root / "txt"
        assert wav_root.is_dir() and txt_root.is_dir()

        # read yaml file as a list of dicts
        with open(txt_root / f"{split}.yaml") as f:
            dicts = yaml.load(f, Loader=yaml.BaseLoader)

        # load source and target utterances and add to dicts
        src = "en"
        tgt = tgt_lang
        for lang in [src, tgt]:
            with open(txt_root / f"{split}.{lang}") as f:
                utts = [r.strip() for r in f]
            assert len(dicts) == len(utts)
            for i, utt in enumerate(utts):
                dicts[i][lang] = utt

        # construct self.data
        self.data = []
        counter = 1
        current_wavfile = ""
        for dict_elem in dicts:
            stem = dict_elem["wav"].split(".")[0]
            uid = f"{stem}_{counter}"
            wav_path = wav_root / f"{uid}.wav"

            self.data.append(
                (wav_path.as_posix(), dict_elem[src], dict_elem[tgt], tgt, uid)
            )

            # assume yaml file is sorted
            if current_wavfile == dict_elem["wav"]:
                counter = 1
            else:
                counter += 1

        # processor to use in __getitem__
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

    def __getitem__(self, idx: int):  # Dict[np.ndarray, torch.Tensor, torch.Tensor]
        """
        Return single input to model
        """
        (
            wav_path,
            src_utt,
            tgt_utt,
            tgt_lang,
            utt_id,
        ) = self.data[idx]
        waveform, sr = sf.read(wav_path)

        sr = torch.tensor([sr])

        with self.processor.as_target_processor():
            # target is src_utt because the task is ASR
            labels = self.processor(
                src_utt, return_tensors="pt", padding=True
            ).input_ids.squeeze()

        return dict(
            waveform=waveform,
            sr=sr,
            labels=labels,
        )

    def __len__(self) -> int:
        return len(self.data)
