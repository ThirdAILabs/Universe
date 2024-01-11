import os
import random
import shutil
from pathlib import Path
from typing import List

import pytest
from ndb_utils import PDF_FILE, all_local_doc_getters, search_works
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit, pytest.mark.release]

ARBITRARY_QUERY = "This is an arbitrary search query"
CHECKPOINT_DIR = "/tmp/neural_db"
NUMBER_DOCS = 1
DOCS_TO_INSERT = [get_doc() for get_doc in all_local_doc_getters[:NUMBER_DOCS]]
OUTPUT_DIM = 1000


def cleanup():
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=False)


def interrupt_immediately(percent_training_completed):
    # This is used to stop the training right after the first 2 batches are trained on.
    if percent_training_completed > 0:
        raise StopIteration("Terminating training at the beginning")


def interrupt_midway(percent_training_completed):
    if percent_training_completed > 0.5:
        raise StopIteration("Terminating training at halfway mark")


def interrupt_at_end(percent_training_completed):
    if percent_training_completed >= 1:
        raise StopIteration("Terminating training at the end")


def train_neural_db_with_checkpoint(number_models: int):
    db = ndb.NeuralDB(
        "user", number_models=number_models, extreme_output_dim=OUTPUT_DIM
    )
    # only training for the first two documents

    checkpoint_config = ndb.CheckpointConfig(
        checkpoint_dir=Path(CHECKPOINT_DIR),
        resume_from_checkpoint=False,
        checkpoint_interval=5,
    )
    db.insert(DOCS_TO_INSERT, train=True, checkpoint_config=checkpoint_config)
    return db


def assert_same_dbs(db1: ndb.NeuralDB, db2: ndb.NeuralDB):
    predictions1 = db1.search(ARBITRARY_QUERY, top_k=5)
    predictions2 = db2.search(ARBITRARY_QUERY, top_k=5)

    assert len(db1.sources()) == len(db2.sources())
    for pred1, pred2 in zip(predictions1, predictions2):
        assert pred1.score == pred2.score
        assert pred1.text == pred2.text


def interrupted_training(number_models: int, interrupt_function):
    # This test first interrupts the training and then resumes it.
    db = ndb.NeuralDB(
        "user", number_models=number_models, extreme_output_dim=OUTPUT_DIM
    )

    checkpoint_config = ndb.CheckpointConfig(
        checkpoint_dir=Path(CHECKPOINT_DIR),
        resume_from_checkpoint=False,
        checkpoint_interval=5,
    )

    try:
        db.insert(
            DOCS_TO_INSERT,
            checkpoint_config=checkpoint_config,
            on_progress=interrupt_function,
        )
    except StopIteration:
        checkpoint_config.resume_from_checkpoint = True
        db.insert(DOCS_TO_INSERT, checkpoint_config=checkpoint_config)
        search_works(db, DOCS_TO_INSERT, assert_acc=True)

        new_db = ndb.NeuralDB.from_checkpoint(
            os.path.join(CHECKPOINT_DIR, "trained.ndb")
        )
        assert_same_dbs(db, new_db)
    except Exception as ex:
        raise ex
    finally:
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
    interrupted_training(number_models=2, interrupt_function=interrupt_immediately)
    interrupted_training(number_models=2, interrupt_function=interrupt_midway)
    interrupted_training(number_models=2, interrupt_function=interrupt_at_end)
