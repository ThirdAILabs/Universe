from thirdai._deps.ColBERT.colbertconfig import ColBERTConfig
from thirdai._deps.ColBERT.colbertmodeling.base_colbert import BaseColBERT

import torch
import string


class ColBERT(BaseColBERT):
    """
    This class handles the basic encoding and scoring operations in ColBERT. It is used for training.
    """

    def __init__(self, name="bert-base-uncased"):
        super().__init__(name)

        if self.colbert_config.mask_punctuation:
            self.skiplist = {
                w: True
                for symbol in string.punctuation
                for w in [
                    symbol,
                    self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0],
                ]
            }

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(
            self.device
        )
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = (
            torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device)
            .unsqueeze(2)
            .float()
        )
        Q = Q * mask

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(
            self.device
        )
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = (
            torch.tensor(
                self.mask(input_ids, skiplist=self.skiplist), device=self.device
            )
            .unsqueeze(2)
            .float()
        )
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == "return_mask":
            return D, mask.bool()

        return D

    def mask(self, input_ids, skiplist):
        mask = [
            [(x not in skiplist) and (x != 0) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask
