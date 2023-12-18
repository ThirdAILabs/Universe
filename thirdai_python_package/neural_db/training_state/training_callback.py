import json
import os
import shutil
from pathlib import Path
from typing import Union

from thirdai import bolt

from ..documents import DocumentDataSource
from ..utils import pickle_to, unpickle_from
from .checkpoint_config import DirectoryConfig
from .training_progress_tracker import NeuralDbProgressTracker


class SaveLoadUtils:
    @staticmethod
    def load_mach_model(path: Path):
        try:
            neural_db_mach_model = unpickle_from(filepath=path)
            return neural_db_mach_model
        except:
            raise Exception(f"Could not find a valid model at the path{path}")

    @staticmethod
    def load_document_data_source(source_path: Path, arguments_path: Path):
        try:
            with open(arguments_path) as f:
                args = json.load(f)
            source = DocumentDataSource.load_from_csv(
                csv_path=source_path,
                id_column=args["id_column"],
                strong_column=args["strong_column"],
                weak_column=args["weak_column"],
            )
        except:
            raise Exception(
                f"Could not load datasource from {source_path}. Ensure that the"
                " checkpoint path is valid."
            )
        return source

    @staticmethod
    def save_mach_model(model, path: Path):
        pickle_to(obj=model, filepath=path)

    @staticmethod
    def save_document_data_source(source, source_path, arguments_path):
        source.save_to_csv(csv_path=source_path)

        with open(arguments_path, "w") as f:
            json.dump(source.initialization_args(), f, indent=4)

    @staticmethod
    def copy_files_to_folder(src_config: DirectoryConfig, tgt_config: DirectoryConfig):
        assert src_config and tgt_config

        source_dir = src_config.checkpoint_dir
        target_dir = tgt_config.checkpoint_dir
        for item in source_dir.iterdir():
            if item.is_file():
                dest_file = target_dir / item.name
                shutil.move(str(item), str(dest_file))


class TrainingProgressManager(bolt.train.callbacks.Callback):  # type: ignore
    def __init__(
        self,
        tracker: NeuralDbProgressTracker,
        dir_config: DirectoryConfig = None,
        neuraldb_mach_model=None,
        intro_source=None,
        train_source=None,
        checkpoint_interval: int = 1,
        is_resumed: bool = False,
    ):
        super().__init__()
        self.tracker = tracker
        self.dir_config = dir_config

        self._neuraldb_mach_model = neuraldb_mach_model
        self._intro_source = intro_source
        self._train_source = train_source
        self.checkpoint_interval = checkpoint_interval
        self._is_resumed = is_resumed

        if self.dir_config:
            self.backup_config = DirectoryConfig(
                checkpoint_dir=self.dir_config.checkpoint_dir / ".temp"
            )
            print(str(self.backup_config.checkpoint_dir))
            os.makedirs(
                os.path.join(self.backup_config.checkpoint_dir),
                exist_ok=True,
            )
        else:
            self.backup_config = None

    def on_epoch_end(self):
        self.tracker.current_epoch_number += 1
        if self.tracker.current_epoch_number % self.checkpoint_interval == 0:
            self.checkpoint_without_sources()

    def on_train_end(self):
        self.checkpoint_without_sources()

    @property
    def neuraldb_mach_model(self):
        if self._neuraldb_mach_model is None:
            self._neuraldb_mach_model = self.load_model()
        return self._neuraldb_mach_model

    @property
    def intro_source(self):
        if self._intro_source is None:
            self._intro_source = self.load_intro_source()
        return self._intro_source

    @property
    def train_source(self):
        if self._train_source is None:
            self._train_source = self.load_train_source()
        return self._train_source

    def load_model(self):
        if self._is_resumed:
            assert self.dir_config != None
            return SaveLoadUtils.load_mach_model(
                path=self.dir_config.neuraldb_model_checkpoint_location
            )

    def load_intro_source(self):
        if self._is_resumed:
            assert self.dir_config != None
            return SaveLoadUtils.load_document_data_source(
                source_path=self.dir_config.intro_source_checkpoint_location,
                arguments_path=self.dir_config.intro_source_arguments_location,
            )

    def load_train_source(self):
        if self._is_resumed:
            assert self.dir_config != None
            return SaveLoadUtils.load_document_data_source(
                source_path=self.dir_config.train_source_checkpoint_location,
                arguments_path=self.dir_config.train_source_arguments_location,
            )

    def checkpoint_model(self):
        if self.neuraldb_mach_model and self.backup_config:
            SaveLoadUtils.save_mach_model(
                self._neuraldb_mach_model,
                self.backup_config.neuraldb_model_checkpoint_location,
            )

    def checkpoint_sources(self):
        if self.backup_config:
            SaveLoadUtils.save_document_data_source(
                source=self.intro_source,
                source_path=self.backup_config.intro_source_checkpoint_location,
                arguments_path=self.backup_config.intro_source_arguments_location,
            )
            SaveLoadUtils.save_document_data_source(
                source=self.train_source,
                source_path=self.backup_config.train_source_checkpoint_location,
                arguments_path=self.backup_config.train_source_arguments_location,
            )
            pickle_to(self.tracker.vlc_config, filepath=self.backup_config.vlc_path)

    def checkpoint_tracker(self):
        if self.backup_config:
            tracker_state = self.tracker.__dict__()
            with open(
                self.backup_config.train_and_intro_state_checkpoint_location, "w"
            ) as f:
                json.dump(tracker_state, f, indent=4)

    def make_preindexing_checkpoint(self):
        if self._is_resumed:
            return
        self.full_checkpoint()
        if self.dir_config:
            SaveLoadUtils.copy_files_to_folder(self.backup_config, self.dir_config)  # type: ignore

    def make_insert_checkpoint(self):
        if self.is_insert_completed:
            return
        self.is_insert_completed = True
        self.checkpoint_without_sources()
        if self.dir_config:
            SaveLoadUtils.copy_files_to_folder(self.backup_config, self.dir_config)  # type: ignore

    def checkpoint_without_sources(self):
        print("making a checkpoint without sources")
        self.checkpoint_model()
        self.checkpoint_tracker()
        if self.dir_config:
            SaveLoadUtils.copy_files_to_folder(self.backup_config, self.dir_config)  # type: ignore

    def full_checkpoint(self):
        print("making a full checkpoint")
        self.checkpoint_model()
        self.checkpoint_sources()
        self.checkpoint_tracker()
        if self.dir_config:
            SaveLoadUtils.copy_files_to_folder(self.backup_config, self.dir_config)  # type: ignore

    def set_pretraining_arguments(
        self,
        fast_approximation: bool,
        num_buckets_to_sample: int,
        override_number_classes: int,
        vlc_config,
    ):
        self.tracker.fast_approximation = fast_approximation
        self.tracker.num_buckets_to_sample = num_buckets_to_sample
        self.tracker.override_number_classes = override_number_classes
        self.tracker.vlc_config = vlc_config

    def set_training_arguments(
        self,
        learning_rate: float,
        min_epochs: int,
        max_epochs: int,
        freeze_before_train: bool,
    ):
        if self.tracker.learning_rate is None:
            self.tracker.learning_rate = learning_rate
        if self.tracker.min_epochs is None:
            self.tracker.min_epochs = min_epochs
        if self.tracker.max_epochs is None:
            self.tracker.max_epochs = max_epochs
        if self.tracker.freeze_before_train is None:
            self.tracker.freeze_before_train = freeze_before_train

    @property
    def is_insert_completed(self):
        return self.tracker.is_insert_completed

    @is_insert_completed.setter
    def is_insert_completed(self, value):
        self.tracker.is_insert_completed = value

    @property
    def is_training_completed(self):
        return self.tracker.is_training_completed

    @is_training_completed.setter
    def is_training_completed(self, value):
        self.tracker.is_training_completed = value

    @property
    def min_epochs(self):
        return self.tracker.min_epochs - self.tracker.current_epoch_number

    @property
    def max_epochs(self):
        return self.tracker.max_epochs - self.tracker.current_epoch_number

    def get_training_arguments(self):
        return {
            "min_epochs": self.min_epochs,
            "max_epochs": self.max_epochs,
            "learning_rate": self.tracker.learning_rate,
            "freeze_before_train": self.tracker.freeze_before_train,
            "max_in_memory_batches": self.tracker.max_in_memory_batches,
            "variable_length": self.tracker.vlc_config,
            "training_progress_manager": self,
        }
