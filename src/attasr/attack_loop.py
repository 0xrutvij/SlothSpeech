import logging
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Callable, Optional, Type

import datasets
import pandas as pd
import torch
from torch.autograd import Variable
from tqdm.auto import tqdm

from attasr.attack_losses import AttackLossProtocol, get_attack_loss
from attasr.experiment_datasets import ExprDataset
from attasr.experiment_models import (
    EXPR_BACKBONE,
    EXPR_FEAT_EXTRACTOR,
    EXPR_TOKENIZER,
    ExprModel,
)
from attasr.noise import get_white_noise

SKIP_TOKENS: dict[ExprModel, set[int]] = {
    ExprModel.Speech2Text: set(),
    ExprModel.Speech2Text2: set(),
    ExprModel.Whisper: {50256, 50257, 50362},
}

NUM_REPS = 2

logger = logging.getLogger()


@dataclass
class EnergyAttackConfig:
    max_iter: int
    learning_rate: float
    optimizer_class: Type[torch.optim.Optimizer]
    adv_dist_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    adv_dist_factor: float
    dataset_name: ExprDataset
    num_datapoints: int
    storage_frequency: int
    storage_location: str
    shard_id: int = -1
    minimize_eos_all_pos: bool = True
    dataset: Optional[datasets.arrow_dataset.Dataset] = None

    def __post_init__(self):
        if not self.dataset:
            self.dataset = ExprDataset.get_dataset(self.dataset_name)

        self.num_datapoints = min(self.num_datapoints, len(self.dataset))
        self.sampling_rate = 16000


@dataclass
class EnergyAttack:
    model: EXPR_BACKBONE
    tokenizer: EXPR_TOKENIZER
    ftex: EXPR_FEAT_EXTRACTOR
    device: str
    skip_tokens: set[int]
    model_name: str
    max_new_tokens: int = 500

    def __post_init__(self):
        self.eos_token_id = self.tokenizer.eos_token_id
        self.model.to(self.device, non_blocking=True)
        self.stats = self._create_stats_table()
        self.tensor_dict = self._create_tensor_dict()

    @classmethod
    def for_model(
        cls,
        expr_model: ExprModel,
        device: str,
        max_new_tokens: int = 500,
    ) -> "EnergyAttack":
        return cls(
            model=ExprModel.get_pretained_model(expr_model),
            tokenizer=ExprModel.get_tokenizer(expr_model),
            ftex=ExprModel.get_feature_extractor(expr_model),
            max_new_tokens=max_new_tokens,
            device=device,
            model_name=expr_model.name,
            skip_tokens=SKIP_TOKENS[expr_model],
        )

    def _create_stats_table(self) -> dict[str, list]:
        return {
            "idx": [],
            "max_token_delta": [],
            "awgn_token_delta": [],
            "og_latency": [],
            "adv_latency": [],
            "awgn_latency": [],
            "perturb_dist": [],
            "awgn_dist": [],
            "og_transcript": [],
            "awgn_transcript": [],
            "og_output_shape": [],
            "adv_output_shape": [],
            "awgn_output_shape": [],
        }

    def _create_tensor_dict(self) -> dict[int, dict[str, Any]]:
        return dict()

    def clear_stats(self):
        self.stats = self._create_stats_table()
        self.tensor_dict = self._create_tensor_dict()

    def get_stats_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.stats)
        df.set_index("idx", inplace=True)
        return df

    def save_stats(self, conf: EnergyAttackConfig, final: bool = False):
        df = self.get_stats_df()
        mname, dname, dcname = (
            self.model_name,
            conf.dataset_name.name,
            conf.adv_dist_criterion.__name__,
        )
        df["model"] = mname
        df["dataset"] = dname
        df["dist_criteria"] = dcname
        sid = str(conf.shard_id) if conf.shard_id != -1 else ""
        save_path_base = f"{mname}_{dname}_{dcname}{sid}"
        save_path = Path(conf.storage_location).joinpath(f"{save_path_base}")
        df.to_csv(save_path.with_suffix(".csv"))
        torch.save(self.tensor_dict, save_path.with_suffix(".pt"))

    def _append_stats(
        self,
        idx: int,
        og_output_shape: tuple[int, ...],
        adv_output_shape: tuple[int, ...],
        max_token_delta: int,
        og_latency: int,
        adv_latency: int,
        perturb_dist: float,
        og_transcript: str,
        longest_transcript: str,
        awgn_token_delta: int,
        awgn_latency: int,
        awgn_dist: float,
        awgn_transcript: str,
        awgn_output_shape: tuple[int, ...],
        raw_perturbation: torch.Tensor,
    ):
        self.stats["idx"].append(idx)
        self.stats["og_output_shape"].append(og_output_shape)
        self.stats["adv_output_shape"].append(adv_output_shape)
        self.stats["max_token_delta"].append(max_token_delta)
        self.stats["og_latency"].append(og_latency)
        self.stats["perturb_dist"].append(perturb_dist)
        self.stats["og_transcript"].append(og_transcript)
        self.stats["awgn_token_delta"].append(awgn_token_delta)
        self.stats["awgn_latency"].append(awgn_latency)
        self.stats["adv_latency"].append(adv_latency)
        self.stats["awgn_dist"].append(awgn_dist)
        self.stats["awgn_transcript"].append(awgn_transcript)
        self.stats["awgn_output_shape"].append(awgn_output_shape)

        self.tensor_dict[idx] = {
            "longest_transcript": longest_transcript,
            "raw_perturbation": raw_perturbation,
        }

    def launch(self, config: EnergyAttackConfig, prog_pos: int = -1) -> None:
        datapoints = tqdm(
            enumerate(config.dataset),  # type: ignore
            total=config.num_datapoints,
            desc="Datapoints",
            position=prog_pos + 1,
        )

        num_proc = 0
        for i, datapoint in datapoints:
            if num_proc >= config.num_datapoints:
                break

            if num_proc > 0 and num_proc % config.storage_frequency == 0:
                self.save_stats(config)

            inputs = self.ftex(
                datapoint["audio"]["array"],
                sampling_rate=config.sampling_rate,
                return_tensors="pt",
            )

            ip_raw = (
                inputs.input_features
                if "input_features" in inputs
                else inputs.input_values
            )

            if (
                self.model_name == ExprModel.Speech2Text2.name
                and ip_raw.shape[1] > 400_000
            ):
                continue

            num_proc += 1
            input_features: torch.Tensor = Variable(
                ip_raw,
                requires_grad=True,
            ).to(self.device, non_blocking=True)

            logger.debug(
                "\nThe size of input features\n", input_features.shape
            )

            modifier_var: torch.Tensor = Variable(
                torch.zeros_like(input_features), requires_grad=True
            ).to(self.device, non_blocking=True)

            optimizer = config.optimizer_class(  # type: ignore
                [modifier_var], lr=config.learning_rate
            )

            time_totals = 0.0
            skip_dp = False
            for _ in range(NUM_REPS):
                start = time()
                try:
                    generated_ids = self.model.generate(
                        inputs=input_features,
                        max_new_tokens=self.max_new_tokens,
                    )
                    time_totals += time() - start
                except Exception:
                    skip_dp = True
                    break

            if skip_dp:
                continue

            og_latency = time_totals / NUM_REPS

            original_output_shape = generated_ids.shape
            original_ouput_length = original_output_shape[1]
            best_adv_x = modifier_var + input_features
            best_output_shape = original_output_shape
            best_perturbation = modifier_var.detach().cpu()
            max_token_delta = float("-inf")

            attack_loss_criterion: AttackLossProtocol = get_attack_loss(
                skip_tokens=self.skip_tokens,
                eos_token_id=self.eos_token_id,
                adv_dist_factor=config.adv_dist_factor,
                adv_dist_criterion=config.adv_dist_criterion,
                minimize_eos_all_pos=config.minimize_eos_all_pos,
            )

            attack_iterations = tqdm(
                range(config.max_iter),
                desc="Attack Iteration",
                position=prog_pos + 2,
            )
            for j in attack_iterations:
                adv_x = modifier_var + input_features
                try:
                    generated_ids_j = self.model.generate(
                        inputs=adv_x, max_new_tokens=self.max_new_tokens
                    )
                except Exception:
                    break

                logits = self.model(
                    adv_x, decoder_input_ids=generated_ids_j
                ).logits
                optimizer.zero_grad()
                loss = attack_loss_criterion(logits, adv_x, input_features)
                loss.backward()
                optimizer.step()

                current_output_shape = generated_ids_j.shape
                current_output_length = generated_ids_j.shape[1]
                token_delta = current_output_length - original_ouput_length
                if token_delta > max_token_delta:
                    best_adv_x = adv_x.detach().cpu()
                    best_output_shape = current_output_shape
                    max_token_delta = token_delta
                    best_perturbation = modifier_var.detach().cpu()
                if int(max_token_delta) >= 500:
                    break

            time_totals = 0
            skip_dp = False
            for _ in range(NUM_REPS):
                start = time()
                try:
                    generated_ids_best = self.model.generate(
                        inputs=best_adv_x.to(self.device),
                        max_new_tokens=self.max_new_tokens,
                    )
                    time_totals += time() - start
                except Exception:
                    skip_dp = True
                    break

            if skip_dp:
                continue

            adv_latency = time_totals / NUM_REPS

            noise = get_white_noise(input_features, signal_to_noise_ratio=10)
            noisy_input = (
                noise.to(self.device, non_blocking=True) + input_features
            )

            time_totals = 0
            skip_dp = False
            for _ in range(NUM_REPS):
                start = time()
                try:
                    generated_ids_noisy = self.model.generate(
                        noisy_input.to(self.device, non_blocking=True),
                        max_new_tokens=self.max_new_tokens,
                    )
                    time_totals += time() - start
                except Exception:
                    skip_dp = True
                    break

            if skip_dp:
                continue

            awgn_latency = time_totals / NUM_REPS
            awgn_distance = (
                torch.nn.functional.l1_loss(
                    noisy_input.to(self.device), input_features
                )
                .detach()
                .cpu()
                .item()
            )

            awgn_output_shape = generated_ids_noisy.shape
            awgn_token_delta = awgn_output_shape[1] - original_ouput_length

            awgn_xscript = self.tokenizer.batch_decode(
                generated_ids_noisy, skip_special_tokens=True
            )[0]

            og_xscript = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            best_xscript = self.tokenizer.batch_decode(
                generated_ids_best, skip_special_tokens=True
            )[0]

            perturb_dist = (
                torch.nn.functional.l1_loss(
                    best_adv_x.to(self.device), input_features
                )
                .detach()
                .cpu()
                .item()
            )

            logger.info(
                f"\n{'-' * 70}"
                f"\nLargest Change: {int(max_token_delta)}"
                f"\nOriginal Transcription: {og_xscript}"
                f"\nLongest Transcription: {best_xscript[:50]}"
                f"\n\n{'-' * 70}\n\n"
            )

            self._append_stats(
                idx=i,
                og_output_shape=tuple(original_output_shape),
                adv_output_shape=tuple(best_output_shape),
                max_token_delta=int(max_token_delta),
                perturb_dist=perturb_dist,
                og_transcript=og_xscript,
                longest_transcript=best_xscript,
                adv_latency=int(adv_latency * 1000),
                og_latency=int(og_latency * 1000),
                awgn_token_delta=awgn_token_delta,
                awgn_latency=int(awgn_latency * 1000),
                awgn_dist=awgn_distance,
                awgn_transcript=awgn_xscript,
                awgn_output_shape=awgn_output_shape,
                raw_perturbation=best_perturbation.flatten().detach().cpu(),
            )

        self.save_stats(conf=config, final=True)
        self.clear_stats()
        return
