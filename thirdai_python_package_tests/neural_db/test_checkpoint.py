import os
import random
import shutil
from pathlib import Path
from typing import List

import pytest
from ndb_utils import PDF_FILE, all_local_doc_getters
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit, pytest.mark.release]
ARBITRARY_QUERY = "This is an arbitrary search query"
CHECKPOINT_DIR = "/tmp/neural_db"


def cleanup():
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=False)


def interrupt_immediately(percent_training_completed):
    # This is used to stop the training right after the first 2 batches are trained on.
    if percent_training_completed > 0:
        raise Exception("Terminating training at the beginning")


def interrupt_midway(percent_training_completed):
    if percent_training_completed > 0.5:
        raise Exception("Terminating training at halfway mark")


def interrupt_at_end(percent_training_completed):
    if percent_training_completed >= 1:
        raise Exception("Terminating training at the end")


def train_neural_db_with_checkpoint(number_models: int):
    db = ndb.NeuralDB("user", number_models=number_models)
    all_docs = [get_doc() for get_doc in all_local_doc_getters]

    checkpoint_config = ndb.NDBCheckpointConfig(
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

    assert correct_source / sum([doc.size for doc in docs]) > 0.7
    if assert_acc:
        assert correct_result / sum([doc.size for doc in docs]) > 0.7


def assert_same_dbs(db1: ndb.NeuralDB, db2: ndb.NeuralDB):
    predictions1 = db1.search(ARBITRARY_QUERY, top_k=5)
    predictions2 = db2.search(ARBITRARY_QUERY, top_k=5)

    assert len(db1.sources()) == len(db2.sources())
    for pred1, pred2 in zip(predictions1, predictions2):
        assert pred1.score == pred2.score
        assert pred1.text == pred2.text


def interrupted_training(number_models: int, interrupt_function):
    # This test first interrupts the training and then resumes it.
    db = ndb.NeuralDB("user", number_models=number_models)
    all_docs = [get_doc() for get_doc in all_local_doc_getters]

    checkpoint_config = ndb.NDBCheckpointConfig(
        checkpoint_dir=Path(CHECKPOINT_DIR),
        resume_from_checkpoint=False,
        checkpoint_interval=1,
    )

    try:
        db.insert(
            all_docs,
            checkpoint_config=checkpoint_config,
            on_progress=interrupt_function,
        )
    finally:
        db = ndb.NeuralDB("user", number_models=number_models)
        checkpoint_config.resume_from_checkpoint = True
        db.insert(all_docs, checkpoint_config=checkpoint_config)
        search_works(db, all_docs, assert_acc=True)

        new_db = ndb.NeuralDB.from_checkpoint(
            os.path.join(CHECKPOINT_DIR, "trained.ndb")
        )
        assert_same_dbs(db, new_db)

    cleanup()


def test_neural_db_checkpoint_on_single_mach():
    db = train_neural_db_with_checkpoint(number_models=1)
    loaded_db = ndb.NeuralDB.from_checkpoint(
        os.path.join(CHECKPOINT_DIR, "trained.ndb")
    )

    assert_same_dbs(db, loaded_db)
    cleanup()


def test_neural_db_checkpoint_on_mach_mixture():
    db = train_neural_db_with_checkpoint(number_models=2)
    loaded_db = ndb.NeuralDB.from_checkpoint(
        os.path.join(CHECKPOINT_DIR, "trained.ndb")
    )
    assert_same_dbs(db, loaded_db)
    cleanup()


def test_interrupted_training_mach():
    interrupted_training(number_models=1, interrupt_function=interrupt_immediately)
    interrupted_training(number_models=1, interrupt_function=interrupt_midway)
    interrupted_training(number_models=1, interrupt_function=interrupt_at_end)


def test_interrupted_training_mach_mixture():
    interrupted_training(number_models=3, interrupt_function=interrupt_immediately)
    interrupted_training(number_models=3, interrupt_function=interrupt_midway)
    interrupted_training(number_models=3, interrupt_function=interrupt_at_end)
