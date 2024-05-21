import math
import random
from typing import List, Tuple

import pandas as pd
from nltk.tokenize import sent_tokenize

from . import utils
from .loggers import Logger
from .models.model_interface import Model


def associate(
    model: Model,
    logger: Logger,
    user_id: str,
    text_pairs: List[Tuple[str, str]],
    top_k: int,
    **kwargs,
):
    model.associate(text_pairs, n_buckets=top_k, **kwargs)
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
    logger: Logger,
    user_id: str,
    query_id_para: List[Tuple[str, int, str]],
    **kwargs,
):
    model.upvote([(query, _id) for query, _id, para in query_id_para], **kwargs)
    logger.log(
        session_id=user_id,
        action="upvote",
        args={"query_id_para": query_id_para},
    )
