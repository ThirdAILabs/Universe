import math
import random
from typing import List, Tuple

import pandas as pd
from nltk.tokenize import sent_tokenize

from . import utils
from .loggers import Logger
from .models import Model


def association_training_samples(
    model: Model, text_a: str, text_b: str, top_k: int, n_samples: int
):
    # Based on Yash's suggestion to chunk target phrase if it is long.
    b_buckets = model.infer_buckets(sent_tokenize(text_b), n_results=top_k)
    samples = [(text_a, buckets) for buckets in b_buckets]
    return utils.random_sample(samples, k=n_samples)


def associate(
    model: Model,
    logger: Logger,
    user_id: str,
    text_pairs: List[Tuple[str, str]],
    top_k: int,
):
    model.associate(text_pairs, top_k)
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
):
    model.upvote([(query, _id) for query, _id, para in query_id_para])
    logger.log(
        session_id=user_id,
        action="upvote",
        args={"query_id_para": query_id_para},
    )
