import os
from typing import Callable, List

import pandas as pd
from docx import Document
from nltk.tokenize import sent_tokenize, word_tokenize

import parsing_utils.utils as utils


def get_elements(filename):
    temp = []
    document = Document(filename)
    prev_short = False
    for p in document.paragraphs:
        if len(p.text.strip()) > 3:
            if prev_short:
                temp[-1] = (temp[-1][0] + " " + p.text.strip(), filename)
            else:
                temp.append((p.text.strip(), filename))
            prev_short = (
                len(word_tokenize(p.text.strip())) < utils.ATTACH_N_WORD_THRESHOLD
            )
    temp = [
        (chunk, filename)
        for passage, filename in temp
        for chunk in utils.chunk_text(passage)
    ]
    return temp, True


def create_train_df(elements):
    df = pd.DataFrame(
        index=range(len(elements)),
        columns=["passage", "para", "filename", "page", "display"],
    )
    for i, elem in enumerate(elements):
        sents = sent_tokenize(str(elem[0]))
        sents = [utils.clean_text(sent).lower() for sent in sents]
        passage = " ".join(sents).replace("\n", " ").strip()
        df.iloc[i] = [
            passage,
            passage,
            elem[1],
            "0",
            str(elem[0].replace("\n", " ")),
        ]
    for column in ["passage", "para", "display"]:
        df[column] = df[column].apply(utils.ensure_valid_encoding)
    return df


def show_docx(PLATFORM, item):
    if PLATFORM == "Windows" or PLATFORM == "win32":
        os.startfile(item)
    else:
        os.system('open "' + item + '"')
