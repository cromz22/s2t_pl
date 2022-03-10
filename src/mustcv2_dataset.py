import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
import yaml
from itertools import groupby
import soundfile as sf

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

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str, str, str]:
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
        ) = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, tgt_lang, utt_id

    def __len__(self) -> int:
        return len(self.data)
