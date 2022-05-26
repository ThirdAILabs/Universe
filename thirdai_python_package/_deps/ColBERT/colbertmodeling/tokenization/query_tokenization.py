import torch

from thirdai._deps.ColBERT.colbertmodeling.hf_colbert import HF_ColBERT
from thirdai._deps.ColBERT.colbertconfig import ColBERTConfig


class QueryTokenizer:
    def __init__(self, config: ColBERTConfig):
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)

        self.config = config
        self.query_maxlen = config.query_maxlen
        self.background_maxlen = (
            512 - self.query_maxlen + 1
        )  # FIXME: Make this configurable

        (
            self.Q_marker_token,
            self.Q_marker_token_id,
        ) = "[Q]", self.tok.convert_tokens_to_ids("[unused0]")
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = (
            self.tok.mask_token,
            self.tok.mask_token_id,
        )

        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [
            prefix
            + lst
            + suffix
            + [self.mask_token] * (self.query_maxlen - (len(lst) + 3))
            for lst in tokens
        ]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):

        ids = self.tok(batch_text, add_special_tokens=False)["input_ids"]

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [
            self.sep_token_id
        ]
        ids = [
            prefix
            + lst
            + suffix
            + [self.mask_token_id] * (self.query_maxlen - (len(lst) + 3))
            for lst in ids
        ]

        return ids

    def tensorize(self, batch_text, context=None):

        # add placehold for the [Q] marker
        batch_text = [". " + x for x in batch_text]

        obj = self.tok(
            batch_text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.query_maxlen,
        )

        ids, mask = obj["input_ids"], obj["attention_mask"]

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        if context is not None:

            obj_2 = self.tok(
                context,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=self.background_maxlen,
            )

            ids_2, mask_2 = (
                obj_2["input_ids"][:, 1:],
                obj_2["attention_mask"][:, 1:],
            )  # Skip the first [SEP]

            ids = torch.cat((ids, ids_2), dim=-1)
            mask = torch.cat((mask, mask_2), dim=-1)

        if self.config.attend_to_mask_tokens:
            mask[ids == self.mask_token_id] = 1

        return ids, mask
