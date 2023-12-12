import json
import os
import pickle
from pathlib import Path

from thirdai import bolt

from .training_progress_tracker import TrainingProgressTracker


def pickle_to(obj: object, filepath: str):
    with open(filepath, "wb") as pkl:
        pickle.dump(obj, pkl)


class TrainingProgressCallback(bolt.train.callbacks.Callback):
    def __init__(self, tracker: TrainingProgressTracker, neuraldb_mach_model=None):
        super().__init__()
        self.tracker = tracker
        self.neuraldb_mach_model = neuraldb_mach_model

        print(
            "making a new directory called"
            f" {os.path.join(tracker.checkpoint_dir, str(tracker.model_id))}"
        )
        os.makedirs(
            os.path.join(tracker.checkpoint_dir, str(tracker.model_id)),
            exist_ok=True,
        )
        self.checkpoint_tracker()

    def on_epoch_end(self):
        self.tracker.current_epoch_number += 1
        self.checkpoint_tracker()

    def on_train_end(self):
        self.checkpoint_tracker()

    def checkpoint_tracker(self):
        print("Checkpointing tracker")
        with open(self.tracker.save_args_checkpoint_location, "w") as f:
            json.dump(json.loads(self.tracker.save_arguments()), f, indent=4)

        with open(self.tracker.intro_args_checkpoint_location, "w") as f:
            json.dump(json.loads(self.tracker.introduce_arguments()), f, indent=4)

        with open(self.tracker.train_args_checkpoint_location, "w") as f:
            json.dump(json.loads(self.tracker.training_arguments()), f, indent=4)

        with open(self.tracker.train_state_checkpoint_location, "w") as f:
            json.dump(json.loads(self.tracker.training_state()), f, indent=4)

        if self.neuraldb_mach_model:
            pickle_to(
                obj=self.neuraldb_mach_model,
                filepath=self.tracker.model_checkpoint_location,
            )
