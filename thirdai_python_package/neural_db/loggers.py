import os
import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd


class Logger:
    def name(self):
        raise NotImplementedError()

    def log(
        self,
        session_id: str,
        action: str,
        args: dict,
        train_samples: Optional[any] = None,
    ):
        raise NotImplementedError()

    def save_meta(self, directory: Path):
        raise NotImplementedError()

    def load_meta(self, directory: Path):
        raise NotImplementedError()

    def save_pkl(self, pkl_file) -> None:
        raise NotImplementedError()

    @staticmethod
    def load_pkl(pkl_data, pkl_file, metadata, metadata_dir) -> None:
        raise NotImplementedError()


class InMemoryLogger(Logger):
    def make_log(session_id=[], action=[], args=[], train_samples=[]):
        return pd.DataFrame(
            {
                "session_id": session_id,
                "action": action,
                "args": args,
                "train_samples": train_samples,
            }
        )

    def __init__(self, logs=make_log()):
        self.logs = logs

    def name(self):
        return "in_memory"

    def log(self, session_id: str, action: str, args: dict, train_samples=None):
        self.logs = pd.concat(
            [
                self.logs,
                InMemoryLogger.make_log(
                    session_id=[session_id],
                    action=[action],
                    args=[args],
                    train_samples=[train_samples],
                ),
            ]
        )

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass

    def save_pkl(self, pkl_file) -> None:
        metadata = {
            "type": "logger",
        }
        pickle.dump(metadata, pkl_file)
        pickle.dump(self, pkl_file)

    @staticmethod
    def load_pkl(pkl_data, pkl_file, metadata, metadata_dir) -> None:
        pass


class LoggerList(Logger):
    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers

    def name(self):
        return "list"

    def log(
        self,
        session_id: str,
        action: str,
        args: dict,
        train_samples: Optional[any] = None,
    ):
        [
            logger.log(
                session_id=session_id,
                action=action,
                args=args,
                train_samples=train_samples,
            )
            for logger in self.loggers
        ]

    def save_meta(self, directory: Path):
        for logger in self.loggers:
            os.mkdir(directory / logger.name())
            logger.save_meta(directory / logger.name())

    def load_meta(self, directory: Path):
        for logger in self.loggers:
            logger.load_meta(directory / logger.name())

    # This variable is needed to not break current load/save
    # We can remove this and all its references if we only use save_pkl/load_pkl
    saving_pkl = False

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the loggers attribute
        if LoggerList.saving_pkl:
            del state["loggers"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Set a default value for loggers since it was not in the state
        if LoggerList.saving_pkl:
            self.loggers = None

    def save_pkl(self, pkl_file):
        LoggerList.saving_pkl = True
        metadata = {"type": "logger", "num_loggers": len(self.loggers)}
        pickle.dump(metadata, pkl_file)
        pickle.dump(self, pkl_file)
        for logger in self.loggers:
            logger.save_pkl(pkl_file)
        LoggerList.saving_pkl = False

    @staticmethod
    def load_pkl(pkl_data, pkl_file, metadata, metadata_dir):
        LoggerList.saving_pkl = True
        logger_list = pkl_data
        loggers = []
        for _ in range(metadata["num_loggers"]):
            logger_metadata = pickle.load(pkl_file)
            logger = pickle.load(pkl_file)
            type(logger).load_pkl(logger, pkl_file, logger_metadata, metadata_dir)
            loggers.append(logger)

        logger_list.loggers = loggers
        LoggerList.saving_pkl = False


class NoOpLogger(Logger):
    def __init__(self) -> None:
        pass

    def name(self):
        return "no_op"

    def log(self, session_id: str, action: str, args: dict, train_samples=None):
        pass

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass

    def save_pkl(self, pkl_file) -> None:
        metadata = {
            "type": "logger",
        }
        pickle.dump(metadata, pkl_file)
        pickle.dump(self, pkl_file)

    @staticmethod
    def load_pkl(pkl_data, pkl_file, metadata, metadata_dir) -> None:
        pass
