from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

import datasets
from datasets import load_dataset


@dataclass
class ExprDatasetProps:
    path: str
    ds_name: Optional[str]
    split: Optional[str]

    def to_dict(self) -> dict:
        dprops = deepcopy(asdict(self))
        dprops["name"] = dprops["ds_name"]
        del dprops["ds_name"]
        return dprops


class ExprDataset(ExprDatasetProps, Enum):
    LibriSpeech = ExprDatasetProps(
        path="patrickvonplaten/librispeech_asr_dummy",
        ds_name="clean",
        split="validation",
    )
    OpenSLR = ExprDatasetProps(
        path="openslr",
        ds_name="SLR70",
        split="train",
    )
    VoxPopuli = ExprDatasetProps(
        path="facebook/voxpopuli",
        ds_name="en",
        split="validation",
    )
    VCTK = ExprDatasetProps(path="vctk", ds_name=None, split=None)

    def __init__(self, data: ExprDatasetProps):
        self.init_args = data.to_dict()

    @staticmethod
    def get_dataset(dataset: "ExprDataset") -> datasets.arrow_dataset.Dataset:
        ds = load_dataset(**dataset.init_args, num_proc=8)
        if dataset.name == ExprDataset.VCTK.name:
            return ds["train"]
        return ds
