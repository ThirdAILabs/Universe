try:
    import nltk

    from . import parsing_utils
    from .constraint_matcher import AnyOf, EqualTo, GreaterThan, InRange, LessThan
    from .documents import (
        CSV,
        DOCX,
        PDF,
        URL,
        Document,
        InMemoryText,
        Reference,
        SalesForce,
        SentenceLevelDOCX,
        SentenceLevelPDF,
        SharePoint,
        SQLDatabase,
        Unstructured,
    )
    from .model_bazaar import Login, ModelBazaar
    from .neural_db import CancelState, CheckpointConfig, NeuralDB, Strength, Sup

    nltk.download("punkt")
except ImportError as error:
    raise ImportError(
        "To use thirdai.neural_db, please install the additional dependencies by"
        " running 'pip install thirdai[neural_db]'"
    )
