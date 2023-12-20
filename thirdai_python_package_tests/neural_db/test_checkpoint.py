import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pytest
import thirdai
from ndb_utils import (
    PDF_FILE,
    all_local_doc_getters,
    create_simple_dataset,
    docs_with_meta,
    metadata_constraints,
    num_duplicate_docs,
    train_simple_neural_db,
)
from thirdai import dataset
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit, pytest.mark.release]
ARBITRARY_QUERY = "This is an arbitrary search query"
CHECKPOINT_DIR = "/tmp/neural_db"


def on_progress(percent_training_completed):
    if percent_training_completed > 0.2:
        raise Exception("Terminating training")


def train_neural_db_with_checkpoint(number_models: int):
    db = ndb.NeuralDB("user", number_models=number_models)
    all_docs = [get_doc() for get_doc in all_local_doc_getters]

    checkpoint_config = ndb.CheckpointConfig(
        checkpoint_dir=Path(CHECKPOINT_DIR),
        resume_from_checkpoint=False,
        checkpoint_interval=1,
    )

    db.insert(all_docs, train=True, checkpoint_config=checkpoint_config)
    return db


def search_works(db: ndb.NeuralDB, docs: List[ndb.Document], assert_acc: bool):
    top_k = 5
    correct_result = 0
    correct_source = 0
    for doc in docs:
        if isinstance(doc, ndb.SharePoint):
            continue
        source = doc.reference(0).source
        for elem_id in range(doc.size):
            query = doc.reference(elem_id).text
            results = db.search(query, top_k)

            assert len(results) >= 1
            assert len(results) <= top_k

            for result in results:
                assert type(result.text) == str
                assert len(result.text) > 0

            correct_result += int(query in [r.text for r in results])
            correct_source += int(source in [r.source for r in results])

    assert correct_source / sum([doc.size for doc in docs]) > 0.8
    if assert_acc:
        assert correct_result / sum([doc.size for doc in docs]) > 0.8


def test_interrupted_training(number_models: int):
    # This test first interrupts the training and then resumes it.
    db = ndb.NeuralDB("user", number_models=number_models)
    all_docs = [get_doc() for get_doc in all_local_doc_getters]

    checkpoint_config = ndb.CheckpointConfig(
        checkpoint_dir=Path(CHECKPOINT_DIR),
        resume_from_checkpoint=False,
        checkpoint_interval=1,
    )

    try:
        db.insert(
            all_docs, checkpoint_config=checkpoint_config, on_progress=on_progress
        )
    finally:
        db = ndb.NeuralDB("user", number_models=number_models)
        checkpoint_config.resume_from_checkpoint = True
        db.insert(all_docs, checkpoint_config=checkpoint_config)
        search_works(db, all_docs, assert_acc=True)


def assert_same_dbs(db1: ndb.NeuralDB, db2: ndb.NeuralDB):
    predictions1 = db1.search(ARBITRARY_QUERY, top_k=5)
    predictions2 = db2.search(ARBITRARY_QUERY, top_k=5)

    assert len(db1.sources()) == len(db2.sources())
    for pred1, pred2 in zip(predictions1, predictions2):
        assert pred1.score == pred2.score
        assert pred1.text == pred2.text


def test_neural_db_checkpoint_on_single_mach():
    db = train_neural_db_with_checkpoint(number_models=1)
    loaded_db = ndb.NeuralDB.from_checkpoint(
        os.path.join(CHECKPOINT_DIR, "trained.ndb")
    )

    assert_same_dbs(db, loaded_db)


def test_neural_db_checkpoint_on_mach_mixture():
    db = train_neural_db_with_checkpoint(number_models=2)
    loaded_db = ndb.NeuralDB.from_checkpoint(
        os.path.join(CHECKPOINT_DIR, "trained.ndb")
    )
    assert_same_dbs(db, loaded_db)
