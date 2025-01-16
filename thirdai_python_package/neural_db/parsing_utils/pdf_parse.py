import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import fitz
import pandas as pd
from nltk.tokenize import sent_tokenize

from .utils import (
    ATTACH_N_WORD_THRESHOLD,
    BlockType,
    chunk_text,
    ensure_valid_encoding,
    get_fitz_text_pages,
)

# TODO: Remove senttokenize
# TODO: Limit paragraph length


@dataclass
class Block:
    x0: float
    y0: float
    x1: float
    y1: float
    lines: str
    block_no: int
    block_type: BlockType

    def __init__(self, block: tuple):
        self.x0 = block[0]
        self.y0 = block[1]
        self.x1 = block[2]
        self.y1 = block[3]
        self.lines = block[4]
        self.block_no = block[5]
        self.block_type = BlockType.Image if block[6] else BlockType.Text


@dataclass
class PDFparagraph:
    text: str
    page_no: int
    filename: str
    block_nums: Union[
        str, Dict[int, List[int]]
    ]  # [Page no. -> Block No(s) Dictionary] in Dict or string format


def para_is_complete(para):
    endings = [".", "?", "!", '."', ".'"]
    return functools.reduce(
        lambda a, b: a or b,
        [para.endswith(end) for end in endings],
    )


# paragraph = {"page_no": [block_id,...], "pagen_no_2":[blicksids, ...]}
def process_pdf_file(
    filepath: str, with_images: bool = False, parallelize: bool = False
):
    try:
        rows = []
        prev = ""
        prev_n_words = float("inf")
        paras = []

        # Get text in fitz block format
        text_pages = get_fitz_text_pages(
            file_path=filepath,
            method="blocks",
            with_images=with_images,
            parallelize=parallelize,
        )

        # sorting to get pages in serial order
        page_numbers = sorted(text_pages.keys())

        for page_no in page_numbers:
            # filtering Blocks depending whether or to parse the images or not.
            blocks = list(
                filter(
                    lambda block: with_images or block.block_type == BlockType.Text,
                    [Block(block) for block in text_pages[page_no]],
                )
            )
            for block in blocks:
                current_block_nums = {}
                current_block_nums[page_no] = [block.block_no]
                current = block.lines.strip()

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
                    prev_blocks = paras[-1].block_nums
                    if page_no in prev_blocks.keys():
                        prev_blocks[page_no].extend(current_block_nums[page_no])
                    else:
                        prev_blocks[page_no] = current_block_nums[page_no]

                    prev_para = paras[-1]
                    prev_para.text += f"\n{current}"
                    prev_para.block_nums = prev_blocks

                else:
                    prev = current
                    paras.append(
                        PDFparagraph(
                            text=current,
                            page_no=page_no,
                            filename=Path(filepath).name,
                            block_nums=current_block_nums,
                        )
                    )

                # Occurrences of space is proxy for number of words.
                # If there are 10 words or less, this paragraph is
                # probably just a header.
                prev_n_words = len(current.split(" "))

        paras = [
            PDFparagraph(
                text=chunk,
                page_no=paragraph.page_no,
                filename=paragraph.filename,
                block_nums=paragraph.block_nums,
            )
            for paragraph in paras
            for chunk in chunk_text(paragraph.text)
        ]
        for para in paras:
            if len(para.text) > 0:
                sent = re.sub(
                    " +",
                    " ",
                    str(para.text).strip(),
                )
                if len(sent) > 0:
                    rows.append(
                        PDFparagraph(
                            text=sent,
                            page_no=para.page_no,
                            filename=para.filename,
                            block_nums=str(para.block_nums),
                        )
                    )
        return rows, True
    except Exception as e:
        print(e.__str__())
        return "Cannot process pdf file:" + filepath, False


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
        df.iloc[i] = [
            para,
            paragraph.filename,
            paragraph.page_no,
            paragraph.text,
            paragraph.block_nums,
        ]
    for column in ["para", "display"]:
        df[column] = df[column].apply(ensure_valid_encoding)
    return df


def highlighted_doc(
    source: str, columns: dict, with_images: bool = False, parallelize: bool = False
):
    if not "highlight" in columns:
        return None
    highlight = eval(columns["highlight"])

    # only extract from the required pages.
    text_pages = get_fitz_text_pages(
        file_path=source,
        method="blocks",
        page_numbers=highlight.keys(),
        with_images=with_images,
        parallelize=parallelize,
    )

    doc = fitz.open(source)
    for page_no, blocks_to_highlight in highlight.items():
        page_blocks = list(
            filter(
                lambda block: with_images or block.block_type == BlockType.Text,
                [Block(block) for block in text_pages[page_no]],
            )
        )
        for block in page_blocks:
            if block.block_no in blocks_to_highlight:
                rect = fitz.Rect(x0=block.x0, y0=block.y0, x1=block.x1, y1=block.y1)
                doc[page_no].add_highlight_annot(rect)
    return doc
