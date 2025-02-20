from importlib import import_module

from scipy.spatial.distance import euclidean
from transformers import AutoTokenizer

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union
import json
import logging

logger = logging.getLogger(__name__)

class ModelType(Enum):
    OT = "ot"
    TS = "ts"
    COCITE = "cocite"

@dataclass
class SimilarityModelConfig:
    """Configuration for TrainedAspire model.

    Attributes:
        base_pt_layer: Pre-trained model identifier
        fine_tune: Whether to enable fine-tuning
        model_version: Version of the model to load
        other_params: Additional model-specific parameters
    """
    base_pt_layer: str
    fine_tune: bool = False
    model_version: str = "cur_best"
    lora:bool = False
    use_bfloat16: bool = False
    attn_implementation: str = "default"  # Options: "default", "flash_attention_2"
    other_params: Optional[Dict] = None

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'SimilarityModelConfig':
        """Create config from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            hparams = data.get('all_hparams', {})
            return cls(
                base_pt_layer=hparams.get('base-pt-layer', ''),
                fine_tune=hparams.get('fine_tune', False),
                lora=hparams.get('lora', False),
                use_bfloat16=hparams.get('use_bfloat16', False),
                attn_implementation=hparams.get('attn_implementation', 'default'),
                other_params=hparams
            )

class ModelFactory:
    """Factory for creating different types of models."""

    _registry = {
        'cocite-specter-biomed-recon': ('src.learning.facetid_models.disent_models.MySPECTER',
                                         'src.learning.batchers.AbsTripleBatcher'),
        'cocite-specter-biomed-specter2': ('src.learning.facetid_models.disent_models.MySPECTER',
                                        'src.learning.batchers.AbsTripleBatcher'),
        'ot-aspire-biomed-recon': ('src.learning.facetid_models.disent_models.WordSentAlignBiEnc',
                                   'src.learning.batchers.AbsSentTokBatcher'),
        'ts-aspire-biomed-recon': ('src.learning.facetid_models.disent_models.WordSentAbsSupAlignBiEnc',
                                   'src.learning.batchers.AbsSentTokBatcherPreAlign'),
        'ts-aspire-biomed-specter2': ('src.learning.facetid_models.disent_models.WordSentAbsSupAlignBiEnc',
                                   'src.learning.batchers.AbsSentTokBatcherPreAlign'),
        'ot-aspire-biomed-specter2': ('src.learning.facetid_models.disent_models.WordSentAlignBiEnc',
                                   'src.learning.batchers.AbsSentTokBatcher'),

        'cocite-gte-qwen2-1.5b-instruct-biomed': ('src.learning.decoder_learning.decoder_models.CoQwen',
                                                    'src.learning.decoder_learning.decoder_batchers.AbsTripleBatcher'),
        'ts-aspire-gte-qwen2-1.5b-instruct-biomed': ('src.learning.decoder_learning.decoder_models.TSQwen',
                                                     'src.learning.decoder_learning.decoder_batchers.AbsSentTokBatcher'),
        'ot-aspire-gte-qwen2-1.5b-instruct-biomed': ('src.learning.decoder_learning.decoder_models.OTQwen',
                                                     'src.learning.decoder_learning.decoder_batchers.AbsSentTokBatcher'),
    }

    @classmethod
    def _import_class(cls, class_path: str) -> Type:
        """Dynamically import a class from a string path.

        Args:
            class_path: String path to the class (e.g., 'disent_models.WordSentAlignBiEnc')

        Returns:
            The imported class
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {class_path}: {str(e)}")

    @classmethod
    def create_model(cls, model_name: str, config: SimilarityModelConfig) -> Tuple:
        """Create model and batcher instances based on model name."""
        if model_name not in cls._registry:
            raise ValueError(f"Unknown model: {model_name}")

        model_path, batcher_path = cls._registry[model_name]

        try:
            model_cls = cls._import_class(model_path)
            batcher_cls = cls._import_class(batcher_path)

            model = model_cls(model_hparams=config.other_params)
            batcher_cls.bert_config_str = config.base_pt_layer

            return model, batcher_cls
        except Exception as e:
            raise RuntimeError(f"Failed to create model {model_name}: {str(e)}")

