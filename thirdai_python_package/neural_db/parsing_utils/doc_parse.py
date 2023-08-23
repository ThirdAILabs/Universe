from pathlib import Path

import pandas as pd
from docx import Document
from nltk.tokenize import sent_tokenize, word_tokenize

from .utils import ATTACH_N_WORD_THRESHOLD, chunk_text, ensure_valid_encoding


def get_elements(filename):
    temp = []
    document = Document(filename)
    prev_short = False
    for p in document.paragraphs:
        if len(p.text.strip()) > 3:
            if prev_short:
                temp[-1] = (temp[-1][0] + " " + p.text.strip(), Path(filename).name)
            else:
                temp.append((p.text.strip(), Path(filename).name))
            prev_short = len(word_tokenize(p.text.strip())) < ATTACH_N_WORD_THRESHOLD
    temp = [
        (chunk, filename) for passage, filename in temp for chunk in chunk_text(passage)
    ]
    return temp, True


def create_train_df(elements):
    df = pd.DataFrame(
        index=range(len(elements)),
        columns=["para", "filename", "display"],
    )
    for i, elem in enumerate(elements):
        sents = sent_tokenize(str(elem[0]))
        sents = [
            sent.replace("\t", " ").replace(",", " ").replace("\n", " ").strip().lower()
            for sent in sents
        ]
        para = " ".join(sents)
        df.iloc[i] = [
            para,
            elem[1],
            str(elem[0].replace("\n", " ")),
        ]
    for column in ["para", "display"]:
        df[column] = df[column].apply(ensure_valid_encoding)
    return df
