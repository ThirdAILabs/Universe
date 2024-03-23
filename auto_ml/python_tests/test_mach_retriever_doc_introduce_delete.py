from mach_retriever_utils import train_simple_mach_retriever, QUERY_FILE
from thirdai import data
import pytest
import pandas as pd

pytestmark = [pytest.mark.unit, pytest.mark.release]


def test_mach_introduce_delete():
    model = train_simple_mach_retriever()

    model.erase(list(range(20)))

    assert model.index.num_entities() == 80

    model.clear()

    assert model.index.num_entities() == 0

    model.introduce(
        data.TransformedIterator(
            data.CsvIterator(QUERY_FILE),
            data.transformations.ToTokens("id", "id"),
        ),
        strong_cols=["text"],
        weak_cols=[],
        text_augmentation=False,
        load_balancing=False,
    )

    assert model.index.num_entities() == 100

    df = pd.read_csv(QUERY_FILE)

    correct = 0
    for i, query in enumerate(df["text"]):
        if model.search(query, top_k=1)[0][0] == i:
            correct += 1

    assert correct / len(df) >= 0.95
