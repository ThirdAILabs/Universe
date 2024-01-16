import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List

import pytest
from ndb_utils import PDF_FILE, all_local_doc_getters, associate_works, upvote_works
from thirdai import data
from thirdai import neural_db as ndb
from thirdai.neural_db.mach_mixture_model import MachMixture
from thirdai.neural_db.models import Mach
from thirdai.neural_db.trainer.training_data_manager import TrainingDataManager
from thirdai.neural_db.trainer.training_progress_manager import TrainingProgressManager
from thirdai.neural_db.trainer.training_progress_tracker import (
    IntroState,
    NeuralDbProgressTracker,
    TrainState,
)
from thirdai.neural_db.utils import pickle_to, unpickle_from

pytestmark = [pytest.mark.unit]

ARBITRARY_QUERY = "This is an arbitrary search query"
CHECKPOINT_DIR = "/tmp/neural_db"
NUMBER_DOCS = 3
DOCS_TO_INSERT = [get_doc() for get_doc in all_local_doc_getters[:NUMBER_DOCS]]
OUTPUT_DIM = 1000


def setup():
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def cleanup():
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)


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


def assert_same_data_sources(source1, source2):
    lines1 = list(source1._get_line_iterator())
    lines2 = list(source2._get_line_iterator())
    for line1, line2 in zip(lines1, lines2):
        assert line1 == line2


def assert_same_dbs(db1: ndb.NeuralDB, db2: ndb.NeuralDB):
    predictions1 = db1.search(ARBITRARY_QUERY, top_k=5)
    predictions2 = db2.search(ARBITRARY_QUERY, top_k=5)

    assert len(db1.sources()) == len(db2.sources())
    for pred1, pred2 in zip(predictions1, predictions2):
        assert pred1.score == pred2.score
        assert pred1.text == pred2.text

    document_source1, document_source2 = (
        db1._savable_state.documents.get_data_source(),
        db2._savable_state.documents.get_data_source(),
    )

    assert_same_data_sources(document_source1, document_source2)


def assert_same_objects(object1, object2):
    for attr in vars(object1).keys():
        assert object1.__getattribute__(attr) == object1.__getattribute__(attr)
    for attr in vars(object2).keys():
        assert object1.__getattribute__(attr) == object1.__getattribute__(attr)


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

        new_db = ndb.NeuralDB.from_checkpoint(
            os.path.join(CHECKPOINT_DIR, "trained.ndb")
        )

        assert_same_dbs(db, new_db)
        upvote_works(db)
        associate_works(db)

    except Exception as ex:
        raise ex
    finally:
        cleanup()


def make_db_and_training_manager(makes_checkpoint=True):
    db = ndb.NeuralDB(number_models=2)
    checkpoint_dir = Path(CHECKPOINT_DIR) / str(0)

    document_manager = db._savable_state.documents
    document_manager.add(DOCS_TO_INSERT)

    save_load_manager = TrainingDataManager(
        checkpoint_dir=checkpoint_dir,
        model=db._savable_state.model.models[0],
        intro_source=document_manager.get_data_source(),
        train_source=document_manager.get_data_source(),
        tracker=NeuralDbProgressTracker(
            IntroState(
                num_buckets_to_sample=8,
                fast_approximation=False,
                override_number_classes=None,
                is_insert_completed=False,
            ),
            train_state=TrainState(
                max_in_memory_batches=None,
                current_epoch_number=0,
                is_training_completed=False,
                learning_rate=0.001,
                min_epochs=5,
                max_epochs=10,
                freeze_before_train=False,
                batch_size=2048,
                freeze_after_epoch=7,
                freeze_after_acc=0.95,
            ),
            vlc_config=data.transformations.VariableLengthConfig(),
        ),
    )

    training_manager = TrainingProgressManager(
        tracker=save_load_manager.tracker,
        save_load_manager=save_load_manager,
        makes_checkpoint=makes_checkpoint,
        checkpoint_interval=3,
    )
    return db, training_manager, checkpoint_dir


# Asserts that the final checkpoint created is the same as the db whose reference is held rn.
@pytest.mark.release
def test_neural_db_checkpoint_on_single_mach():
    db = train_neural_db_with_checkpoint(number_models=1)
    loaded_db = ndb.NeuralDB.from_checkpoint(
        os.path.join(CHECKPOINT_DIR, "trained.ndb")
    )

    assert_same_dbs(db, loaded_db)
    cleanup()


@pytest.mark.release
def test_neural_db_checkpoint_on_mach_mixture():
    db = train_neural_db_with_checkpoint(number_models=2)
    loaded_db = ndb.NeuralDB.from_checkpoint(
        os.path.join(CHECKPOINT_DIR, "trained.ndb")
    )
    assert_same_dbs(db, loaded_db)
    cleanup()


@pytest.mark.release
def test_interrupted_training_single_mach():
    interrupted_training(number_models=1, interrupt_function=interrupt_immediately)
    interrupted_training(number_models=1, interrupt_function=interrupt_midway)
    interrupted_training(number_models=1, interrupt_function=interrupt_at_end)


@pytest.mark.release
def test_interrupted_training_mach_mixture():
    interrupted_training(number_models=2, interrupt_function=interrupt_immediately)
    interrupted_training(number_models=2, interrupt_function=interrupt_midway)
    interrupted_training(number_models=2, interrupt_function=interrupt_at_end)


@pytest.mark.release
def test_reset_mach_model():
    model1 = Mach(
        id_col="id",
        id_delimiter=",",
        query_col="query",
        fhr=50_000,
        embedding_dimension=2048,
        extreme_output_dim=1000,
        extreme_num_hashes=16,
        tokenizer="words",
        hidden_bias=True,
    )

    model2 = Mach()

    for attr in vars(model2).keys():
        model2.__setattr__(attr, 1000)

    model2.reset_model(model1)

    assert_same_objects(model1, model2)


@pytest.mark.release
def test_meta_save_load_for_mach_mixture():
    # This test asserts that the label to segment map is saved and loaded correctly.
    setup()

    model1 = MachMixture(
        number_models=3,
        id_col="id",
        id_delimiter=",",
        query_col="query",
        fhr=50_000,
        embedding_dimension=2048,
        extreme_output_dim=1000,
        extreme_num_hashes=16,
    )

    label_to_segment_map = defaultdict(list)
    label_to_segment_map[0] = 1
    label_to_segment_map[2] = [3, 4]

    model2 = MachMixture(
        number_models=3,
        id_col="id",
        id_delimiter=",",
        query_col="query",
        fhr=50_000,
        embedding_dimension=2048,
        extreme_output_dim=1000,
        extreme_num_hashes=16,
        label_to_segment_map=label_to_segment_map,
        seed_for_sharding=1,
    )

    model2.save_meta(Path(CHECKPOINT_DIR))

    model1.load_meta(Path(CHECKPOINT_DIR))

    assert_same_objects(model1, model2)

    cleanup()


def test_vlc_save_load():
    setup()
    vlc_config = data.transformations.VariableLengthConfig(
        covering_min_length=10, covering_max_length=20, slice_min_length=12
    )
    pickle_to(vlc_config, Path(CHECKPOINT_DIR) / "config.vlc")

    unpickled_config = unpickle_from(Path(CHECKPOINT_DIR) / "config.vlc")

    assert str(vlc_config) == str(unpickled_config)
    cleanup()


def test_tracker_save_load():
    setup()
    tracker = NeuralDbProgressTracker(
        IntroState(
            num_buckets_to_sample=8,
            fast_approximation=False,
            override_number_classes=None,
            is_insert_completed=False,
        ),
        train_state=TrainState(
            max_in_memory_batches=None,
            current_epoch_number=0,
            is_training_completed=False,
            learning_rate=0.001,
            min_epochs=5,
            max_epochs=10,
            freeze_before_train=False,
            batch_size=2048,
            freeze_after_epoch=7,
            freeze_after_acc=0.95,
        ),
        vlc_config=data.transformations.VariableLengthConfig(),
    )

    tracker.save(Path(CHECKPOINT_DIR) / "tracker")
    new_tracker = NeuralDbProgressTracker.load(Path(CHECKPOINT_DIR) / "tracker")

    assert_same_objects(tracker._intro_state, new_tracker._intro_state)
    assert_same_objects(tracker._train_state, tracker._train_state)
    assert str(new_tracker.vlc_config) == str(tracker.vlc_config)
    cleanup()


@pytest.mark.release
def test_save_load_training_data_manager():
    setup()
    _, training_manager, _ = make_db_and_training_manager(makes_checkpoint=True)

    save_load_manager = training_manager.save_load_manager

    save_load_manager.save()

    new_manager = TrainingDataManager.load(Path(CHECKPOINT_DIR) / str(0))

    assert_same_objects(save_load_manager.model, new_manager.model)
    assert_same_objects(
        save_load_manager.tracker._intro_state, new_manager.tracker._intro_state
    )
    assert_same_objects(
        save_load_manager.tracker._train_state, new_manager.tracker._train_state
    )

    assert_same_data_sources(save_load_manager.intro_source, new_manager.intro_source)
    assert_same_data_sources(save_load_manager.train_source, new_manager.train_source)

    cleanup()


@pytest.mark.release
def test_training_progress_manager_no_checkpointing():
    setup()

    _, training_manager, checkpoint_dir = make_db_and_training_manager(
        makes_checkpoint=False
    )

    def assert_no_checkpoints(save_load_manager):
        assert len(os.listdir(save_load_manager.intro_source_folder)) < 1
        assert len(os.listdir(save_load_manager.train_source_folder)) < 1
        assert len(os.listdir(save_load_manager.tracker_folder)) < 1

    training_manager.make_preindexing_checkpoint()
    assert not training_manager.tracker.is_insert_completed
    assert not training_manager.tracker.is_training_completed
    assert_no_checkpoints(training_manager.save_load_manager)

    training_manager.insert_complete()
    assert training_manager.tracker.is_insert_completed
    assert not training_manager.tracker.is_training_completed
    assert_no_checkpoints(training_manager.save_load_manager)

    training_manager.training_complete()
    assert training_manager.tracker.is_insert_completed
    assert training_manager.tracker.is_training_completed
    assert_no_checkpoints(training_manager.save_load_manager)

    cleanup()


@pytest.mark.release
def test_training_progress_manager_with_resuming():
    db, training_manager, checkpoint_dir = make_db_and_training_manager(
        makes_checkpoint=True
    )

    training_manager.make_preindexing_checkpoint()

    resume_training_manager = TrainingProgressManager.from_checkpoint(
        original_mach_model=db._savable_state.model.models[0],
        checkpoint_config=ndb.CheckpointConfig(
            checkpoint_dir=checkpoint_dir,
            resume_from_checkpoint=True,
            checkpoint_interval=1,
        ),
    )

    assert_same_data_sources(
        training_manager.intro_source, resume_training_manager.intro_source
    )
    assert_same_data_sources(
        training_manager.train_source, resume_training_manager.train_source
    )
    assert_same_objects(
        training_manager.save_load_manager.model,
        resume_training_manager.save_load_manager.model,
    )
