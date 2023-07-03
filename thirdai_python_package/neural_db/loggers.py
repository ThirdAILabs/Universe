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
    
    def size(self) -> int:
        raise NotImplementedError()
    
    def filter_by_action(self, action: str) -> pd.DataFrame:
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
        # TODO (Geordie): This doesn't seem very efficient. There must be a 
        # better way to do this.
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
    
    def size(self) -> int:
        return len(self.logs)
    
    def filter_by_action(self, action: str) -> pd.DataFrame:
        return self.logs[self.logs["action"] == action]

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
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
    
    def size(self) -> int:
        return max([logger.size() for logger in self.loggers])
    
    def filter_by_action(self, action: str) -> pd.DataFrame:
        max_logger = NoOpLogger()
        max_size = self.size()
        for logger in self.loggers:
            if logger.size() == max_size:
                max_logger = logger
        
        return max_logger.filter_by_action(action)

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

    def size(self) -> int:
        return 0
    
    def filter_by_action(self, action: str) -> pd.DataFrame:
        return pd.DataFrame({
            "session_id": [],
            "action": [],
            "args": [],
            "train_samples": [],
        })

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass
