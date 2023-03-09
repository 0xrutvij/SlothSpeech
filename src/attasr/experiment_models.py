from dataclasses import dataclass
from enum import Enum
from typing import Type, TypeAlias

from transformers import (
    Speech2Text2Processor,
    Speech2Text2Tokenizer,
    Speech2TextFeatureExtractor,
    Speech2TextForConditionalGeneration,
    Speech2TextTokenizer,
    SpeechEncoderDecoderModel,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

EXPR_BACKBONE: TypeAlias = (
    Speech2TextForConditionalGeneration | WhisperForConditionalGeneration
)

EXPR_TOKENIZER: TypeAlias = Speech2TextTokenizer | WhisperTokenizer

EXPR_FEAT_EXTRACTOR: TypeAlias = (
    Speech2TextFeatureExtractor | WhisperFeatureExtractor
)


@dataclass
class ExprModelProps:
    checkpoint: str
    model_class: Type[EXPR_BACKBONE]
    model_tokenizer: Type[EXPR_TOKENIZER]
    model_feature_extractor: Type[EXPR_FEAT_EXTRACTOR]

    def __hash__(self) -> int:
        return hash(self.checkpoint + self.model_class.__name__)


class ExprModel(ExprModelProps, Enum):
    Speech2Text = ExprModelProps(
        checkpoint="facebook/s2t-small-librispeech-asr",
        model_class=Speech2TextForConditionalGeneration,
        model_tokenizer=Speech2TextTokenizer,
        model_feature_extractor=Speech2TextFeatureExtractor,
    )

    Whisper = ExprModelProps(
        checkpoint="openai/whisper-small.en",
        model_class=WhisperForConditionalGeneration,
        model_tokenizer=WhisperTokenizer,
        model_feature_extractor=WhisperFeatureExtractor,
    )

    Speech2Text2 = ExprModelProps(
        checkpoint="facebook/s2t-wav2vec2-large-en-de",
        model_class=SpeechEncoderDecoderModel,
        model_tokenizer=Speech2Text2Tokenizer,
        model_feature_extractor=Speech2Text2Processor,
    )

    def __init__(self, data: ExprModelProps):
        self.checkpoint = data.checkpoint
        self.model_class = data.model_class
        self.model_tokenizer = data.model_tokenizer
        self.model_feature_extractor = data.model_feature_extractor

    @staticmethod
    def get_pretained_model(model: "ExprModel") -> EXPR_BACKBONE:
        return model.model_class.from_pretrained(model.checkpoint)

    @staticmethod
    def get_tokenizer(model: "ExprModel") -> EXPR_TOKENIZER:
        return model.model_tokenizer.from_pretrained(model.checkpoint)

    @staticmethod
    def get_feature_extractor(model: "ExprModel") -> EXPR_FEAT_EXTRACTOR:
        return model.model_feature_extractor.from_pretrained(model.checkpoint)
