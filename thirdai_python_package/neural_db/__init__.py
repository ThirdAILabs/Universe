try:
    from .documents import CSV, DOCX, PDF, URL, Document, Reference
    from .neural_db import NeuralDB, Strength, Sup
except ImportError as error:
    raise ImportError(
        "To use thirdai.neural_db, please install the additional dependencies by running 'pip install thirdai[neural_db]'"
    )
