import os
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

    def get_logs(self) -> pd.DataFrame:
        raise NotImplementedError()

    def save_meta(self, directory: Path):
        raise NotImplementedError()

    def load_meta(self, directory: Path):
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

    def get_logs(self) -> pd.DataFrame:
        return self.logs

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass


class LoggerList(Logger):
    def __init__(self, loggers: List[Logger]):
        self.loggers = list(
            filter(
                lambda logger: not isinstance(logger, (NoOpLogger, LoggerList)), loggers
            )
        )

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

    def get_logs(self):
        if len(self.loggers) == 0:
            return pd.DataFrame(
                {
                    "session_id": [],
                    "action": [],
                    "args": [],
                    "train_samples": [],
                }
            )
        return self.loggers[0].get_logs()

    def save_meta(self, directory: Path):
        for logger in self.loggers:
            os.mkdir(directory / logger.name())
            logger.save_meta(directory / logger.name())

    def load_meta(self, directory: Path):
        for logger in self.loggers:
            logger.load_meta(directory / logger.name())


class NoOpLogger(Logger):
    def __init__(self) -> None:
        pass

    def name(self):
        return "no_op"

    def log(self, session_id: str, action: str, args: dict, train_samples=None):
        pass

    def get_logs(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "session_id": [],
                "action": [],
                "args": [],
                "train_samples": [],
            }
        )

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass
