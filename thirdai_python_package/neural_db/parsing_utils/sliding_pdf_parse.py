import fitz
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import unidecode


def get_fitz_blocks(filename):
    doc = fitz.open(filename)
    return [
        {**block, "page_num": num}
        for num, page in enumerate(doc)
        for block in page.get_text("dict")["blocks"]
    ]


def remove_images(blocks):
    return [block for block in blocks if block["type"] == 0]


def get_text_len(block):
    return sum(len(span["text"]) for line in block["lines"] for span in line["spans"])


def remove_header_footer(blocks):
    # https://github.com/pymupdf/PyMuPDF/discussions/2259#discussioncomment-6669190
    dbscan = DBSCAN()
    samples = np.array([(*block["bbox"], get_text_len(block)) for block in blocks])
    dbscan.fit(samples)
    labels = dbscan.labels_
    label_counter = Counter(labels)
    most_common_label = label_counter.most_common(1)[0][0]
    return [block for block, label in zip(blocks, labels) if label == most_common_label]


def remove_nonstandard_orientation(blocks):
    orient_to_count = defaultdict(lambda: 0)
    for block in blocks:
        for line in block["lines"]:
            orient_to_count[line["dir"]] += 1
    #
    standard_orientation = sorted(
        [(count, orient) for orient, count in orient_to_count.items()]
    )[-1][1]
    #
    return [
        {
            **block,
            "lines": [
                line for line in block["lines"] if line["dir"] == standard_orientation
            ],
        }
        for block in blocks
    ]


def strip_spaces(blocks):
    return [
        {
            **block,
            "lines": [
                {
                    **line,
                    "spans": [
                        {**span, "text": span["text"].strip()} for span in line["spans"]
                    ],
                }
                for line in block["lines"]
            ],
        }
        for block in blocks
    ]


def remove_empty_spans(blocks):
    return [
        {
            **block,
            "lines": [
                {
                    **line,
                    "spans": [span for span in line["spans"] if len(span["text"]) > 0],
                }
                for line in block["lines"]
            ],
        }
        for block in blocks
    ]


def remove_empty_lines(blocks):
    return [
        {**block, "lines": [line for line in block["lines"] if len(line["spans"]) > 0]}
        for block in blocks
    ]


def remove_empty_blocks(blocks):
    blocks = strip_spaces(blocks)
    blocks = remove_empty_spans(blocks)
    blocks = remove_empty_lines(blocks)
    return [block for block in blocks if len(block["lines"]) > 0]


def get_lines(blocks):
    return [
        {**line, "page_num": block["page_num"]}
        for block in blocks
        for line in block["lines"]
    ]


def set_line_text(lines):
    """Assums span texts are stripped of leading and trailing whitespaces."""
    return [
        {**line, "text": " ".join(span["text"] for span in line["spans"])}
        for line in lines
    ]


def set_line_word_counts(lines):
    """Assums line texts are set"""
    return [{**line, "word_count": line["text"].count(" ") + 1} for line in lines]


def get_lines_with_first_n_words(lines, n):
    total_len = 0
    end = 0
    while total_len < n:
        total_len += lines[end]["word_count"]
        end += 1
    return " ".join(line["text"] for line in lines[:end])


def get_chunks_from_lines(lines, chunk_words, stride_words):
    chunk_start = 0
    chunks = []
    chunk_boxes = []
    while chunk_start < len(lines):
        chunk_end = chunk_start
        chunk_size = 0
        while chunk_size < chunk_words and chunk_end < len(lines):
            chunk_size += lines[chunk_end]["word_count"]
            chunk_end += 1
        stride_end = chunk_start
        stride_size = 0
        while stride_size < stride_words and stride_end < len(lines):
            stride_size += lines[stride_end]["word_count"]
            stride_end += 1
        chunks.append(" ".join(line["text"] for line in lines[chunk_start:chunk_end]))
        chunk_boxes.append(
            (line["page_num"], line["bbox"]) for line in lines[chunk_start:chunk_end]
        )
        chunk_start = stride_end
    return chunks, chunk_boxes


def clean_encoding(text):
    return unidecode.unidecode(text.encode("utf-8", "replace").decode("utf-8"))


def get_chunks(filename, chunk_words, stride_words, emphasize_first_n_words):
    blocks = get_fitz_blocks(filename)
    blocks = remove_images(blocks)
    blocks = remove_header_footer(blocks)
    blocks = remove_nonstandard_orientation(blocks)
    blocks = remove_empty_blocks(blocks)
    lines = get_lines(blocks)
    lines = set_line_text(lines)
    lines = set_line_word_counts(lines)
    emphasis = get_lines_with_first_n_words(lines, emphasize_first_n_words)
    chunks, chunk_boxes = get_chunks_from_lines(lines, chunk_words, stride_words)
    chunks = [clean_encoding(text) for text in chunks]
    return chunks, chunk_boxes, emphasis


def make_df(filename, chunk_words, stride_words, emphasize_first_n_words):
    chunks, chunk_boxes, emphasis = get_chunks(
        filename, chunk_words, stride_words, emphasize_first_n_words
    )
    return pd.DataFrame(
        {
            "para": [c.lower() for c in chunks],
            "display": chunks,
            "emphasis": [emphasis for _ in chunks],
            # chunk_boxes is a list of (page_num, bbox) pairs
            "highlight": str(chunk_boxes),
        }
    )
