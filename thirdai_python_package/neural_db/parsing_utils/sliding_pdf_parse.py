from collections import Counter, defaultdict

import fitz
import numpy as np
import pandas as pd
import pdfplumber
import unidecode
from sklearn.cluster import DBSCAN


def get_fitz_blocks(filename):
    doc = fitz.open(filename)
    blocks = [
        {**block, "page_num": num}
        for num, page in enumerate(doc)
        for block in page.get_text("dict")["blocks"]
    ]
    doc.close()
    return blocks


def remove_images(blocks):
    return [block for block in blocks if block["type"] == 0]


def get_text_len(block):
    return sum(len(span["text"]) for line in block["lines"] for span in line["spans"])


def remove_header_footer(blocks):
    # Inspired by
    # https://github.com/pymupdf/PyMuPDF/discussions/2259#discussioncomment-6669190
    # TL;DR:
    # 1. Vectorize each block by their rectangle coordinates + length of text
    # 2. Use DBSCAN to cheaply cluster these vectors
    # 3. Only headers and footers are repetitive enough to form its own
    # cluster, so they will have a unique label while the majority of blocks are
    # classified as clusterless.
    # 4. Remove blocks that belong to minority clusters.
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
    sorted_count_orient_pairs = sorted(
        [(count, orient) for orient, count in orient_to_count.items()]
    )
    _count, most_frequent_orientation = sorted_count_orient_pairs[-1]
    #
    return [
        {
            **block,
            "lines": [
                line
                for line in block["lines"]
                if line["dir"] == most_frequent_orientation
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
    """Assumes span texts are stripped of leading and trailing whitespaces."""
    return [
        {**line, "text": " ".join(span["text"] for span in line["spans"])}
        for line in lines
    ]


def set_line_word_counts(lines):
    """Assumes line texts are set"""
    return [{**line, "word_count": line["text"].count(" ") + 1} for line in lines]


def get_lines_with_first_n_words(lines, n):
    total_len = 0
    end = 0
    while total_len < n:
        total_len += lines[end]["word_count"]
        end += 1
    return " ".join(line["text"] for line in lines[:end])


def estimate_section_titles(lines):
    text_size_freq = defaultdict(int)
    for line in lines:
        for span in line["spans"]:
            text_size_freq[int(span["size"])] += 1
    most_common_text_size = sorted(text_size_freq.items(), key=lambda x: x[1])[-1][0]
    current_section_title = ""
    for i in range(len(lines)):
        if lines[i]["spans"][0]["size"] > most_common_text_size:
            current_section_title = lines[i]["text"]
        lines[i]["section_title"] = current_section_title
    return lines


def process_tables(filename, lines):
    pdf = pdfplumber.open(filename)
    tables_by_page = [page.find_tables() for page in pdf.pages]

    def within_by_bbox(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        def within(val, lower, upper):
            return lower < val and val < upper

        return (
            within(ax1, bx1, bx2)
            and within(ax2, bx1, bx2)
            and within(ay1, by1, by2)
            and within(ay2, by1, by2)
        )

    new_lines = []
    seen = defaultdict(bool)
    for line in lines:
        is_contained_by_table = False
        for table in tables_by_page[line["page_num"]]:
            if within_by_bbox(line["bbox"], table.bbox):
                is_contained_by_table = True
                if not seen[table]:
                    # The format we want the table in for cold_start is different
                    # than the format we'd like to display so we modify the "text"
                    # field and we include some display metadata in the line
                    # Chunking of text within tables is left for future work
                    extracted_rows = table.extract()
                    table_as_text = " ".join(
                        [item or "" for row in extracted_rows for item in row]
                    )
                    new_lines.append(
                        {
                            **line,
                            "text": table_as_text,
                            "bbox": table.bbox,
                            "table": extracted_rows,
                        }
                    )
                    seen[table] = True
        if not is_contained_by_table:
            new_lines.append(line)

    return new_lines


def get_chunks_from_lines(lines, chunk_words, stride_words):
    chunk_start = 0
    chunks = []
    chunk_boxes = []
    section_titles = []
    display = []
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
        # combine the chunks
        chunks.append(" ".join(line["text"] for line in lines[chunk_start:chunk_end]))
        # metadata for highlighting
        chunk_boxes.append(
            [(line["page_num"], line["bbox"]) for line in lines[chunk_start:chunk_end]]
        )
        # combine unique section titles
        if "section_title" in lines[0]:
            unique_section_titles = set(
                [line["section_title"] for line in lines[chunk_start:chunk_end]]
            )
            section_titles.append(" ".join(unique_section_titles))
        # formulate display text (separating out tables)
        display.append(
            " ".join(
                [
                    str(line["table"]) if "table" in line else line["text"]
                    for line in lines[chunk_start:chunk_end]
                ]
            )
        )
        chunk_start = stride_end
    return chunks, chunk_boxes, display, section_titles


def clean_encoding(text):
    return unidecode.unidecode(text.encode("utf-8", "replace").decode("utf-8"))


def get_chunks(
    filename,
    chunk_words,
    stride_words,
    emphasize_first_n_words,
    ignore_header_footer,
    ignore_nonstandard_orientation,
    emphasize_section_titles,
    table_parsing,
):
    blocks = get_fitz_blocks(filename)
    blocks = remove_images(blocks)
    if ignore_header_footer:
        blocks = remove_header_footer(blocks)
    if ignore_nonstandard_orientation:
        blocks = remove_nonstandard_orientation(blocks)
    blocks = remove_empty_blocks(blocks)
    lines = get_lines(blocks)
    lines = set_line_text(lines)
    if emphasize_section_titles:
        lines = estimate_section_titles(lines)
    if table_parsing:
        lines = process_tables(filename, lines)
    lines = set_line_word_counts(lines)
    first_n_words = get_lines_with_first_n_words(lines, emphasize_first_n_words)
    chunks, chunk_boxes, display, section_titles = get_chunks_from_lines(
        lines, chunk_words, stride_words
    )
    chunks = [clean_encoding(text) for text in chunks]
    section_titles = [clean_encoding(section_title) for section_title in section_titles]
    return chunks, chunk_boxes, display, first_n_words, section_titles


def make_df(
    filename,
    chunk_words,
    stride_words,
    emphasize_first_n_words,
    ignore_header_footer,
    ignore_nonstandard_orientation,
    doc_keywords,
    emphasize_section_titles,
    table_parsing,
):
    """Arguments:
    chunk_size: number of words in each chunk of text.
    stride: number of words between each chunk of text.
    emphasize_first_words: number of words at the beginning of the file
        that will be used as strong column for all rows of the resulting
        dataframe. We do this so that every row can capture important signals
        like file titles or introductory paragraphs.
    table_parsing: Whether to enable separate parsing of tables
    """
    chunks, chunk_boxes, display, first_n_words, section_titles = get_chunks(
        filename,
        chunk_words,
        stride_words,
        emphasize_first_n_words,
        ignore_header_footer,
        ignore_nonstandard_orientation,
        emphasize_section_titles,
        table_parsing,
    )

    emphasis = [
        (
            first_n_words
            + " "
            + doc_keywords
            + " "
            + (section_titles[i] if emphasize_section_titles else "")
        )
        for i in range(len(chunks))
    ]

    return pd.DataFrame(
        {
            "para": [c.lower() for c in chunks],
            "display": display,
            "emphasis": emphasis,
            # chunk_boxes is a list of lists of (page_num, bbox) pairs
            "chunk_boxes": [str(chunk_box) for chunk_box in chunk_boxes],
            # get the first element of the first pair in the list of
            # (page_num, bbox) pairs for each chunk.
            "page": [chunk_box[0][0] for chunk_box in chunk_boxes],
        }
    )


def highlighted_doc(source, columns):
    if not "chunk_boxes" in columns:
        return None
    doc = fitz.open(source)
    for page, box in eval(columns["chunk_boxes"]):
        doc[page].add_highlight_annot(fitz.Rect(box))
    return doc
