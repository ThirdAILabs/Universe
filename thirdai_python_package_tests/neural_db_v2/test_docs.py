import os
from typing import List

import pandas as pd
import pytest
from ndbv2_utils import (
    CSV_FILE,
    DOCX_FILE,
    EML_FILE,
    PDF_FILE,
    PPTX_FILE,
    TXT_FILE,
    URL_LINK,
)
from thirdai.neural_db_v2 import (
    CSV,
    DOCX,
    PDF,
    PPTX,
    URL,
    Document,
    Email,
    InMemoryText,
    TextFile,
)

pytestmark = [pytest.mark.unit, pytest.mark.release]


def all_empty_strings(series: pd.Series):
    return series.map(lambda x: isinstance(x, str) and len(x) == 0).all()


def all_strings(series: pd.Series):
    return series.map(lambda x: isinstance(x, str)).any()


def all_nonempty_strings(series: pd.Series):
    return series.map(lambda x: isinstance(x, str) and len(x) > 0).all()


def doc_property_checks(
    doc: Document,
    has_keywords: bool,
    document_metadata: dict,
    chunk_metadata_columns: List[str],
    allow_empty_keywords: bool = False,
):
    has_chunks = False
    for chunks in doc.chunks():
        assert len(chunks.text) > 0
        assert all_nonempty_strings(chunks.text)

        if has_keywords:
            if allow_empty_keywords:
                assert all_strings(chunks.keywords)
            else:
                assert all_nonempty_strings(chunks.keywords)
        else:
            assert all_empty_strings(chunks.keywords)
        assert len(chunks.keywords) == len(chunks.text)

        assert len(chunks.document) == len(chunks.text)
        assert all_nonempty_strings(chunks.document)
        assert chunks.document.nunique() == 1

        if len(document_metadata) > 0:
            for key, value in document_metadata.items():
                assert key in chunks.metadata.columns
                unique = chunks.metadata[key].unique()
                assert len(unique) == 1
                assert unique[0] == value

        for col in chunk_metadata_columns:
            assert col in chunks.metadata.columns

        if len(document_metadata) + len(chunk_metadata_columns) > 0:
            assert len(chunks.metadata) == len(chunks.text)
            assert len(chunks.metadata.columns) == len(
                set(chunk_metadata_columns).union(document_metadata.keys())
            )
        else:
            assert chunks.metadata is None

        has_chunks = True

    assert has_chunks


def test_csv_doc_infered_columns():
    df = pd.read_csv(CSV_FILE)

    doc = CSV(CSV_FILE)
    doc_property_checks(
        doc,
        has_keywords=False,
        document_metadata={},
        chunk_metadata_columns=["category"],
    )
    chunks = list(doc.chunks())[0]
    assert (df["text"] == chunks.text).all()


def test_csv_doc_no_keywords():
    df = pd.read_csv(CSV_FILE)

    doc = CSV(CSV_FILE, text_columns=["text", "text"], doc_metadata={"type": "csv"})
    doc_property_checks(
        doc,
        has_keywords=False,
        document_metadata={"type": "csv"},
        chunk_metadata_columns=["category"],
    )
    chunks = list(doc.chunks())[0]
    assert ((df["text"] + " " + df["text"]) == chunks.text).all()
    assert (df["category"] == chunks.metadata["category"]).all()
    assert (chunks.metadata["type"] == "csv").all()


def test_csv_doc_with_keywords():
    df = pd.read_csv(CSV_FILE)

    doc = CSV(
        CSV_FILE,
        text_columns=["text"],
        keyword_columns=["text", "text"],
        doc_metadata={"type": "csv"},
    )
    doc_property_checks(
        doc,
        has_keywords=True,
        document_metadata={"type": "csv"},
        chunk_metadata_columns=["category"],
    )
    chunks = list(doc.chunks())[0]
    assert (df["text"] == chunks.text).all()
    assert ((df["text"] + " " + df["text"]) == chunks.keywords).all()
    assert (chunks.metadata["type"] == "csv").all()


def test_csv_doc_streaming():
    df = pd.read_csv(CSV_FILE)

    doc = CSV(
        CSV_FILE,
        text_columns=["text"],
        keyword_columns=["text", "text"],
        doc_metadata={"type": "csv"},
        max_rows=3,
    )
    doc_property_checks(
        doc,
        has_keywords=True,
        document_metadata={"type": "csv"},
        chunk_metadata_columns=["category"],
    )

    part1, part2 = list(doc.chunks())

    all_texts = pd.concat([part1.text, part2.text]).to_list()
    assert df["text"].to_list() == all_texts

    all_keywords = pd.concat([part1.keywords, part2.keywords]).to_list()
    assert (df["text"] + " " + df["text"]).to_list() == all_keywords

    assert (pd.concat([part1.metadata, part2.metadata])["type"] == "csv").all()


@pytest.mark.parametrize("metadata", [{}, {"val": "abc"}])
def test_docx_doc(metadata):
    doc = DOCX(DOCX_FILE, doc_metadata=metadata)

    doc_property_checks(
        doc=doc,
        has_keywords=False,
        document_metadata=metadata,
        chunk_metadata_columns=[],
    )


@pytest.mark.parametrize("metadata", [{}, {"val": "abc"}])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_pdf_doc(metadata, version):
    doc = PDF(PDF_FILE, version=version, doc_metadata=metadata)

    doc_property_checks(
        doc=doc,
        has_keywords=True,
        document_metadata=metadata,
        chunk_metadata_columns=(
            ["chunk_boxes", "page"] if version == "v2" else ["highlight", "page"]
        ),
        allow_empty_keywords=True,
    )


@pytest.mark.parametrize("metadata", [{}, {"val": "abc"}])
def test_email_doc(metadata):
    doc = Email(EML_FILE, doc_metadata=metadata)

    doc_property_checks(
        doc=doc,
        has_keywords=False,
        document_metadata=metadata,
        chunk_metadata_columns=["filetype", "subject", "sent_from", "sent_to"],
    )


@pytest.mark.parametrize("metadata", [{}, {"val": "abc"}])
def test_pptx_doc(metadata):
    doc = PPTX(PPTX_FILE, doc_metadata=metadata)

    doc_property_checks(
        doc=doc,
        has_keywords=False,
        document_metadata=metadata,
        chunk_metadata_columns=["filetype", "page"],
    )


@pytest.mark.parametrize("metadata", [{}])
def test_txt_doc(metadata):
    doc = TextFile(TXT_FILE, doc_metadata=metadata)

    doc_property_checks(
        doc=doc,
        has_keywords=False,
        document_metadata=metadata,
        chunk_metadata_columns=["filetype"],
    )


@pytest.mark.parametrize("metadata", [{}, {"val": "abc"}])
def test_url_doc(metadata):
    for title_is_strong in [True, False]:
        doc = URL(url=URL_LINK, title_is_strong=title_is_strong, doc_metadata=metadata)

        doc_property_checks(
            doc=doc,
            has_keywords=True,
            document_metadata=metadata,
            chunk_metadata_columns=[],
        )


@pytest.mark.parametrize("metadata", [{}, {"val": "abc"}])
def test_in_memory_text_doc(metadata):

    doc = InMemoryText(
        document_name="test",
        text=["a b", "c d"],
        chunk_metadata=[{"item": 1}, {"item": 2}],
        doc_metadata=metadata,
    )

    doc_property_checks(
        doc=doc,
        has_keywords=False,
        document_metadata=metadata,
        chunk_metadata_columns=["item"],
    )

    chunks = doc.chunks()[0]

    assert (chunks.text == pd.Series(["a b", "c d"])).all()
    assert (chunks.metadata["item"] == pd.Series([1, 2])).all()
