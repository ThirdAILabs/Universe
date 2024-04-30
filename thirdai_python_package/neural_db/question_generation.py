from typing import List

from thirdai.gen.questions import QAGenMethod

from .documents import Document
from .supervised_datasource import Sup


def gen_questions(documents: List[Document], generator: QAGenMethod):
    sups = []
    for doc in documents:
        texts = [doc.reference(i).text for i in range(doc.size)]
        generated_questions = []
        labels = []
        for i, questions in enumerate(generator.generate(texts)):
            generated_questions.extend(questions)
            labels.extend([[i] for _ in range(len(questions))])
        sups.append(Sup(queries=generated_questions, labels=labels, source_id=doc.hash))
    return sups
