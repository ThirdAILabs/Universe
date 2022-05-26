import torch

# from transformers import BertTokenizerFast

from thirdai._deps.ColBERT.colbertmodeling.hf_colbert import HF_ColBERT
from thirdai._deps.ColBERT.colbertconfig import ColBERTConfig


class DocTokenizer:
    def __init__(self, config: ColBERTConfig):
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)

        self.config = config
        self.doc_maxlen = config.doc_maxlen

        (
            self.D_marker_token,
            self.D_marker_token_id,
        ) = "[D]", self.tok.convert_tokens_to_ids("[unused1]")
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id

    def tokenize(self, batch_text, add_special_tokens=False):

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):

        ids = self.tok(batch_text, add_special_tokens=False)["input_ids"]

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.D_marker_token_id], [
            self.sep_token_id
        ]
        ids = [prefix + lst + suffix for lst in ids]

        return ids

    def tensorize(self, batch_text):

        # add placehold for the [D] marker
        batch_text = [". " + x for x in batch_text]

        obj = self.tok(
            batch_text,
            padding="longest",
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.doc_maxlen,
        )

        ids, mask = obj["input_ids"], obj["attention_mask"]

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        return ids, mask
