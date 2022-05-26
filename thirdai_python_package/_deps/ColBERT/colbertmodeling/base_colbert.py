import os
import torch

from thirdai._deps.ColBERT.colbertutils.utils import torch_load_dnn

from transformers import AutoTokenizer
from thirdai._deps.ColBERT.colbertmodeling.hf_colbert import HF_ColBERT
from thirdai._deps.ColBERT.colbertconfig import ColBERTConfig


class BaseColBERT(torch.nn.Module):
    """
    Shallow module that wraps the ColBERT parameters, custom configuration, and underlying tokenizer.
    This class provides direct instantiation and saving of the model/colbert_config/tokenizer package.

    Like HF, evaluation mode is the default.
    """

    def __init__(self, name):
        super().__init__()

        self.name = name
        self.colbert_config = ColBERTConfig.load_from_checkpoint(name)
        self.model = HF_ColBERT.from_pretrained(
            name, colbert_config=self.colbert_config
        )
        self.raw_tokenizer = AutoTokenizer.from_pretrained(self.model.base)

        self.eval()

    @property
    def device(self):
        return self.model.device

    @property
    def bert(self):
        return self.model.bert

    @property
    def linear(self):
        return self.model.linear

    @property
    def score_scaler(self):
        return self.model.score_scaler
