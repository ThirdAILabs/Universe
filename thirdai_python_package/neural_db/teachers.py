import math
import random
from typing import List, Optional, Tuple

import pandas as pd
from nltk.tokenize import sent_tokenize

from . import utils
from .inverted_index import InvertedIndex
from .loggers import Logger
from .models.models import Model


def associate(
    model: Model,
    inverted_index: Optional[InvertedIndex],
    logger: Logger,
    user_id: str,
    text_pairs: List[Tuple[str, str]],
    top_k: int,
    **kwargs,
):
    model.associate(text_pairs, n_buckets=top_k, **kwargs)
    if inverted_index:
        inverted_index.associate(text_pairs)
    logger.log(
        session_id=user_id,
        action="associate",
        args={
            "pairs": text_pairs,
            "top_k": top_k,
        },
    )


def upvote(
    model: Model,
    inverted_index: Optional[InvertedIndex],
    logger: Logger,
    user_id: str,
    query_id_para: List[Tuple[str, int, str]],
    **kwargs,
):
    pairs = [(query, _id) for query, _id, para in query_id_para]
    model.upvote(pairs, **kwargs)
    if inverted_index:
        inverted_index.upvote(pairs)
    logger.log(
        session_id=user_id,
        action="upvote",
        args={"query_id_para": query_id_para},
    )
