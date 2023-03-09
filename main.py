import logging
import warnings
from datetime import datetime
from pathlib import Path
from time import time

import torch

from attasr.attack_loop import EnergyAttack, EnergyAttackConfig
from attasr.attack_losses import l2_norm, linf_norm
from attasr.experiment_datasets import ExprDataset
from attasr.experiment_models import ExprModel

warnings.filterwarnings("ignore")

CUDA_DEVICE = "cuda:3"
MODEL = "whisper"  # "s2t", "s2t2"
DATASET = "libri"  # "vctk", "openslr"
DIST_CRITERION = "l2"  # "linf"

DATASET_DICT = {
    "libri": ExprDataset.LibriSpeech,
    "vctk": ExprDataset.VCTK,
    "openslr": ExprDataset.OpenSLR,
}

MODEL_DICT = {
    "whisper": ExprModel.Whisper,
    "s2t2": ExprModel.Speech2Text2,
    "s2t": ExprModel.Speech2Text,
}

DIST_CRITERIA = {"l2": l2_norm, "linf": linf_norm}

logging.basicConfig(
    filename=(
        f"logs/{MODEL.capitalize()}{DATASET.capitalize()}"
        f"{DIST_CRITERION.capitalize()}.log"
    ),
    filemode="w",
    level=logging.INFO,
)

logger = logging.getLogger()

try:
    start = time()
    dataset_type = DATASET_DICT[DATASET]
    dataset = ExprDataset.get_dataset(dataset_type)

    load_time = time() - start

    logger.info(
        "Completed loading the dataset in "
        f"{load_time//60:.0f}m {load_time%60:.0f}s\n"
    )

    model_type = MODEL_DICT[MODEL]
    attack_base = EnergyAttack.for_model(model_type, device=CUDA_DEVICE)

    logger.info(
        f"Completed instantiating the attack for {MODEL} in "
        f"{load_time//60:.0f}m {load_time%60:.0f}s\n"
    )

    dist_criterion = DIST_CRITERIA[DIST_CRITERION]
    conf = EnergyAttackConfig(
        max_iter=101,
        learning_rate=1e-1,
        optimizer_class=torch.optim.Adam,
        adv_dist_criterion=dist_criterion,
        adv_dist_factor=0.1,
        num_datapoints=100,
        dataset_name=dataset_type,
        dataset=dataset,
        storage_frequency=20,
        storage_location=str(Path("out").absolute()),
    )

    logger.info(
        f"Starting Attacks... "
        f"{datetime.now().isoformat(timespec='seconds', sep=' ')}"
    )
    attack_base.launch(conf)
except Exception as e:
    logger.exception(e.args)
