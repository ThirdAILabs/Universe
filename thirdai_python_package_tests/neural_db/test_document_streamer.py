import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from kafka import KafkaConsumer, KafkaProducer
from ndb_utils import CSV_FILE, all_local_doc_getters
from thirdai import neural_db as ndb

pytest = [pytest.mark.unit]

TOPIC = "test"
HOST = "localhost"
PORT = 9092


@pytest.fixture(scope="session")
def produce_data():
    data = pd.read_csv(CSV_FILE).to_dict(orient="records")
    producer = KafkaProducer(
        bootstrap_servers=f"{HOST}:{PORT}",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    for v in data:
        producer.send(topic=TOPIC, value=v)
    producer.flush()
    producer.close()


@pytest.fixture(scope="function")
def create_kafka_doc():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=f"{HOST}:{PORT}",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )

    return ndb.Kafka(
        consumer=consumer,
        topic_name=TOPIC,
        id_column="category",
        strong_columns=["text"],
        weak_columns=["text"],
    )


def test_doc_property(produce_data, create_kafka_doc):
    kafka_doc = create_kafka_doc()

    assert isinstance(kafka_doc.name, str)
    assert isinstance(kafka_doc.hash, str)
    assert isinstance(kafka_doc.source, str)


def test_save_load_method(produce_data, create_kafka_doc):
    kafka_doc = create_kafka_doc()
    saved_path = "doc_save_dir"
    kafka_doc.save(saved_path)

    loaded_doc = ndb.Document.load(saved_path)
    shutil.rmtree(saved_path)
    with pytest.raises(AttributeError):
        streamer = getattr(loaded_doc, loaded_doc._get_streamer_object_name)


def test_doc_equivalence(produce_data, create_kafka_doc):
    kafka_doc = create_kafka_doc()
    csv_doc = all_local_doc_getters[0]

    for row in kafka_doc.row_iterator():
        assert csv_doc.strong_text(row.id) == row.strong
        assert csv_doc.weak_text(row.id) == row.weak
