import functools
import re
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum

import fitz
import pandas as pd
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Union

from .utils import ATTACH_N_WORD_THRESHOLD, chunk_text, ensure_valid_encoding

# TODO: Remove senttokenize
# TODO: Limit paragraph length

class BlockType(IntEnum):
    Text = 0
    Image = 1
    
@dataclass
class Block():
    x0: float
    y0: float
    x1: float
    y1: float
    lines: str
    block_no: int
    block_type: BlockType

@dataclass
class PDFparagraph():
    text: str
    page_no: int
    filename: str
    block_nos: Union[str, Dict[int, List[int]]]     # [Page no. -> Block No(s) Dictionary] in Dict or string format

def para_is_complete(para):
    endings = [".", "?", "!", '."', ".'"]
    return functools.reduce(
        lambda a, b: a or b,
        [para.endswith(end) for end in endings],
    )


# paragraph = {"page_no": [block_id,...], "pagen_no_2":[blicksids, ...]}
def process_pdf_file(filename):
    try:
        rows = []
        prev = ""
        prev_n_words = float("inf")
        doc = fitz.open(filename)
        paras = []
        for page_no, page in enumerate(doc):
            blocks = page.get_text("blocks")
            for t in blocks:
                block = Block(x0 = t[0], y0=t[1], x1=t[2], y1=t[3], lines=t[4], block_no = t[5], block_type=t[6])
                if block.block_type == BlockType.Text:
                    current_blocks = {}
                    current_blocks[page_no] = [block.block_no]
                    current = sent_tokenize(
                        block.lines.strip().replace("\r\n", " ").replace("\n", " ")
                    )
                    current = " ".join(current)

                    if (
                        len(paras) > 0
                        and prev != ""
                        and (
                            not para_is_complete(paras[-1].text)
                            or prev_n_words < ATTACH_N_WORD_THRESHOLD
                        )
                    ):
                        attach = True
                    else:
                        attach = False

                    if attach and len(paras) > 0:
                        prev_blocks = paras[-1].block_nos
                        if page_no in prev_blocks.keys():
                            prev_blocks[page_no].extend(current_blocks[page_no])
                        else:
                            prev_blocks[page_no] = current_blocks[page_no]
                            
                        prev_para = paras[-1]
                        prev_para.text += f" {current}"
                        prev_para.block_nos = prev_blocks

                    else:
                        prev = current
                        paras.append(PDFparagraph(text=current,
                                               page_no=page_no,
                                               filename=Path(filename).name,
                                               block_nos=current_blocks))

                    # Occurrences of space is proxy for number of words.
                    # If there are 10 words or less, this paragraph is
                    # probably just a header.
                    prev_n_words = len(current.split(" "))

        paras = [
            PDFparagraph(text = chunk, page_no=paragraph.page_no, filename = paragraph.filename, block_nos = paragraph.block_nos)
            for paragraph in paras
            for chunk in chunk_text(paragraph.text)
        ]
        for para in paras:
            if len(para.text) > 0:
                sent = re.sub(
                    " +",
                    " ",
                    str(para.text)
                    .replace("\t", " ")
                    .replace(",", " ")
                    .replace("\n", " ")
                    .strip(),
                )
                if len(sent) > 0:
                    rows.append(PDFparagraph(text = sent, page_no=para.page_no, filename=para.filename, block_nos=str(para.block_nos)))
        return rows, True
    except Exception as e:
        print(e.__str__())
        return "Cannot process pdf file:" + filename, False


def create_train_df(elements):
    df = pd.DataFrame(
        index=range(len(elements)),
        columns=["para", "filename", "page", "display", "highlight"],
    )
    for i, paragraph in enumerate(elements):
        sents = sent_tokenize(paragraph.text)
        sents = list(map(lambda x: x.lower(), sents))
        para = " ".join(sents)
        # elem[-1] is id
        df.iloc[i] = [para, paragraph.filename, paragraph.page_no, paragraph.text, paragraph.block_nos]
    for column in ["para", "display"]:
        df[column] = df[column].apply(ensure_valid_encoding)
    return df
