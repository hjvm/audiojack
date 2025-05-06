import json
from typing import List, Dict, Union, Any

import torch
from torch.utils.data import Dataset
import torchaudio

__all__ = [
    "time_to_frame",
    "SyllableDataset",
]

SAMPLE_RATE: int = 16_000   # HuBERT expects 16 kHz audio
FRAME_STRIDE: int = 320     # 20 ms to 320 samples at 16 kHz

def time_to_frame(t: float, sample_rate: int = SAMPLE_RATE, frame_stride: int = FRAME_STRIDE) -> int:
    """Convert time (seconds) to HuBERT conv-frame index (int)."""
    return int(t * sample_rate // frame_stride)


class SyllableDataset(Dataset):
    """A minimal PyTorch `Dataset` for syllable-level training.

    Parameters
    ----------
    data : Union[str, List[Dict]]
        *path to a jsonl* (one json per line) or
        a Python list whose elements already match the manifest schema.
    sample_rate : int, optional
        Target sampling rate after (re)sampling.  Default 16 kHz.
    frame_stride : int, optional
        Conv stride in samples (HuBERT uses 320).  Default 320.
    lazy : bool, optional
        When *True* (default) audio is loaded on-the-fly in `__getitem__`.
        Set *False* if to preload audio.
    """

    def __init__(
        self,
        data: Union[str, List[Dict[str, Any]]],
        sample_rate: int = SAMPLE_RATE,
        frame_stride: int = FRAME_STRIDE,
        lazy: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(data, str):
            with open(data, "r", encoding="utf-8") as fp:
                self.items: List[Dict[str, Any]] = [json.loads(l) for l in fp]
        else:
            self.items = list(data)  # shallow copy

        self.sample_rate = sample_rate
        self.frame_stride = frame_stride
        self.lazy = lazy

        if not self.lazy:
            for itm in self.items:
                wav, sr = torchaudio.load(itm["file"])
                if sr != self.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
                itm["waveform"] = wav

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        itm = self.items[idx]

        if self.lazy or "waveform" not in itm:
            wav, sr = torchaudio.load(itm["file"])
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        else:
            wav = itm["waveform"]  # pre‑cached

        bounds = []
        labels = []
        for start_t, end_t, syl_id in itm["info"]:
            s = time_to_frame(start_t, self.sample_rate, self.frame_stride)
            e = max(time_to_frame(end_t, self.sample_rate, self.frame_stride) - 1, s)
            bounds.append([s, e])
            labels.append(syl_id)
        

        return {
            "waveform": wav,                                            
            "syllable_bounds": torch.tensor(bounds, dtype=torch.long), 
            "labels": torch.tensor(labels, dtype=torch.float)           
        }

