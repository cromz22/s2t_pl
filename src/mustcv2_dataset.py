import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
import yaml
from itertools import groupby
import soundfile as sf
from transformers import Wav2Vec2Processor

from utils.audio_utils import get_waveform


class MuSTCv2Dataset(Dataset):
    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    TGT_LANGUAGES = ["de", "zh", "ja"]

    def __init__(self, root: str, tgt_lang: str, split: str) -> None:
        assert split in self.SPLITS and tgt_lang in self.TGT_LANGUAGES
        _root = Path(root) / f"en-{tgt_lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()

        # Load audio segments
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)

        # Load source and target utterances
        src = "en"
        tgt = tgt_lang
        for lang in [src, tgt]:
            with open(txt_root / f"{split}.{lang}") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][lang] = u

        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment[src],
                        segment[tgt],
                        segment["speaker_id"],
                        tgt,
                        _id,
                    )
                )

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

    def __getitem__(self, idx: int):  # Dict[np.ndarray, torch.Tensor, torch.Tensor]
        """
        single input
        """
        (
            wav_path,
            offset,
            n_frames,
            sr,
            src_utt,
            tgt_utt,
            spk_id,
            tgt_lang,
            utt_id,
        ) = self.data[idx]
        # waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        # waveform = torch.from_numpy(waveform).squeeze()
        # return waveform, sr, src_utt, tgt_utt, spk_id, tgt_lang, utt_id
        waveform, sr = sf.read(wav_path)

        sr = torch.tensor([sr])

        with self.processor.as_target_processor():
            # target is src_utt because the task is ASR
            labels = self.processor(src_utt, return_tensors="pt", padding=True).input_ids.squeeze()

        return dict(
            waveform=waveform,
            sr=sr,
            labels=labels,
        )

    def __len__(self) -> int:
        return len(self.data)
