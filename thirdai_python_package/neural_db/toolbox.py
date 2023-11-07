import uuid
from pathlib import Path

import pandas as pd
from thirdai._thirdai import bolt

from .constraint_matcher import AnyOf
from .documents import PDF
from .neural_db import NeuralDB
from .lexical_utils import reformulate as lexical_reformulate
from .lexical_utils import rerank as lexical_rerank
from .model_bazaar import Bazaar
from .constraint_matcher import AnyOf
from .lexical_utils import reformulate as lexical_reformulate, rerank as lexical_rerank
import uuid
import os


def pdf_file_model(files, in_dim=50_000, emb_dim=2048, num_buckets=50_000, epochs=10):
    dfs = [PDF(file).df for file in files]
    dfs = [
        pd.DataFrame(
            {
                "para": df["para"],
                "doc_id": [i for _ in range(len(df))],
                "file": [Path(files[i]).name for _ in range(len(df))],
            }
        )
        for i, df in enumerate(dfs)
    ]
    file_level_coldstart = f"__file_level_cs_{uuid.uuid4()}__.csv"
    pd.concat(dfs).to_csv(file_level_coldstart, index=False)
    #
    udt = bolt.UniversalDeepTransformer(
        data_types={
            "query": bolt.types.text(tokenizer="char-4"),
            "doc_id": bolt.types.categorical(delimiter=" "),
        },
        target="doc_id",
        n_target_classes=len(files),
        integer_target=True,
        options={
            "extreme_classification": True,
            "extreme_output_dim": num_buckets,
            "fhr": in_dim,
            "embedding_dimension": emb_dim,
            "rlhf": True,
        },
    )
    udt.cold_start(
        file_level_coldstart,
        strong_column_names=[],
        weak_column_names=["para"],
        learning_rate=0.005,
        epochs=epochs,
    )
    os.remove(file_level_coldstart)
    #
    for df in dfs:
        df["para"].iloc[0] = "\n".join(df["para"])
    one_reference_per_file_df = pd.concat([df.iloc[:1] for df in dfs])
    ndb_reference_file = f"__ndb_reference_file_{uuid.uuid4()}__.csv"
    one_reference_per_file_df.to_csv(ndb_reference_file, index=False)
    return NeuralDB.from_udt(
        udt,
        csv=ndb_reference_file,
        csv_id_column="doc_id",
        csv_strong_columns=[],
        csv_weak_columns=["para"],
        csv_reference_columns=["para"],
    )


def pdf_para_model(files, bazaar_cache):
    if not os.path.exists(bazaar_cache):
        os.mkdir(bazaar_cache)
    bazaar = Bazaar(cache_dir=Path(bazaar_cache))
    bazaar.fetch()
    para_db = bazaar.get_model("General QnA")
    docs = [PDF(file, metadata={"file": Path(file).name}) for file in files]
    para_db.insert(docs, train=True)
    return para_db


def hierarchical_search(
    file_db,
    para_db,
    query,
    top_k_returned,
    top_k_files=5,
    top_k_rerank=100,
    rerank=True,
    reformulate=False,
):
    file_results = file_db.search(
        query=query,
        top_k=top_k_files,
    )
    top_k_filenames = [r.metadata["file"] for r in file_results]
    constraints = {"file": AnyOf(top_k_filenames)}
    return rerank_and_reformulate(
        para_db,
        query,
        top_k_returned=top_k_returned,
        top_k_rerank=top_k_rerank,
        rerank=rerank,
        reformulate=reformulate,
        constraints=constraints,
    )


def rerank_and_reformulate(
    db,
    query,
    top_k_returned,
    top_k_rerank=100,
    rerank=True,
    reformulate=False,
    constraints={},
):
    results = db.search(query=query, top_k=top_k_rerank, constraints=constraints)
    if reformulate:
        results = lexical_reformulate(db, query, constraints=constraints)
    if rerank:
        results = lexical_rerank(query, results)
    return results[:top_k_returned]
