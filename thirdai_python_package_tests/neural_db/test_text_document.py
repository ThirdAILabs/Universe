import os
import shutil
import pytest
import pickle
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit]


def test_text_document_different_text_different_hash():
    text1 = ndb.Text(name="text", texts=["text 1", "text 2"])
    text2 = ndb.Text(name="text", texts=["text 3", "text 4"])
    assert text1.hash != text2.hash


def test_text_document_different_metadata_different_hash():
    text1 = ndb.Text(
        name="text", texts=["text 1", "text 2"], metadatas=[{"a": 1}, {"a": 2}]
    )
    text2 = ndb.Text(
        name="text", texts=["text 1", "text 2"], metadatas=[{"a": 3}, {"a": 4}]
    )
    assert text1.hash != text2.hash


def test_text_document_same_contents_same_hash():
    text1 = ndb.Text(
        name="text", texts=["text 1", "text 2"], metadatas=[{"a": 1}, {"a": 2}]
    )
    text2 = ndb.Text(
        name="text", texts=["text 1", "text 2"], metadatas=[{"a": 1}, {"a": 2}]
    )
    assert text1.hash == text2.hash


def test_text_document_texts_are_weak_column():
    text = ndb.Text(name="text", texts=["text 1", "text 2"])
    assert text.weak_text(0) == "text 1"
    assert text.weak_text(1) == "text 2"
    assert text.strong_text(0) == ""
    assert text.strong_text(1) == ""


def test_text_document_filters_entity_ids_by_constraint():
    text = ndb.Text(
        name="text",
        texts=["text 1", "text 2"],
        metadatas=[{"a": 1}, {"a": 2}],
        global_metadata={"b": 1},
    )
    assert text.filter_entity_ids({"a": ndb.EqualTo(0)}) == []
    assert text.filter_entity_ids({"a": ndb.EqualTo(1)}) == [0]
    assert text.filter_entity_ids({"a": ndb.EqualTo(2)}) == [1]
    assert text.matched_constraints["b"].value() == 1


def test_text_document_returns_references_with_text_and_metadata():
    text = ndb.Text(
        name="text",
        texts=["text 1", "text 2"],
        metadatas=[{"a": 1}, {"a": 2}],
        global_metadata={"b": 1},
    )
    assert text.reference(0).text == "text 1"
    assert text.reference(0).metadata["a"] == 1
    assert text.reference(0).metadata["b"] == 1
    assert text.reference(1).text == "text 2"
    assert text.reference(1).metadata["a"] == 2
    assert text.reference(1).metadata["b"] == 1


def test_text_document_serialization():
    text = ndb.Text(
        name="text",
        texts=["text 1", "text 2"],
        metadatas=[{"a": 1}, {"a": 2}],
        global_metadata={"b": 1},
    )
    with open("text_doc.pkl", "wb") as pkl:
        pickle.dump(text, pkl)
    meta = "./text_meta_temp"
    if not os.path.exists(meta):
        os.mkdir(meta)
    text.save_meta(meta)
    with open("text_doc.pkl", "rb") as pkl:
        unpickled_text = pickle.load(pkl)
    unpickled_text.load_meta(meta)
    shutil.rmtree(meta)
    assert unpickled_text.reference(0).text == "text 1"
    assert unpickled_text.reference(0).metadata["a"] == 1
    assert unpickled_text.reference(0).metadata["b"] == 1
    assert unpickled_text.reference(1).text == "text 2"
    assert unpickled_text.reference(1).metadata["a"] == 2
    assert unpickled_text.reference(1).metadata["b"] == 1
