import time

import pandas as pd
import tqdm
from thirdai import bolt


def calculate_precision(all_relevant_documents, all_recommended_documents, at=1):
    assert len(all_relevant_documents) == len(all_recommended_documents)

    precision = 0
    for relevant_documents, recommended_documents in zip(
        all_relevant_documents, all_recommended_documents
    ):
        score = 0
        for pred_doc_id in recommended_documents[:at]:
            if pred_doc_id in relevant_documents:
                score += 1
        score /= at
        precision += score

    return precision / len(all_recommended_documents)


def get_relevant_documents(supervised_tst_file):
    relevant_documents = []
    with open(supervised_tst_file, "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            try:
                query, doc_ids = line.split(",")
            except:
                doc_ids, query, text = line.split(",")
            doc_ids = [int(doc_id.strip()) for doc_id in doc_ids.split(":")]
            relevant_documents.append(doc_ids)

    return relevant_documents


def evaluate_model(model, supervised_tst, use_sparse_inference=False):
    test_df = pd.read_csv(supervised_tst)
    test_samples = [{"QUERY": text} for text in test_df["QUERY"].tolist()]

    elapsed = 0.0
    all_recommended_documents = []
    for i in tqdm.tqdm(range(0, len(test_samples), 10000)):
        start = time.perf_counter()
        output = model.predict_batch(
            test_samples[i : i + 10000], sparse_inference=use_sparse_inference
        )
        end = time.perf_counter()

        elapsed += end - start

        for sample in output:
            all_recommended_documents.append([int(doc) for doc, score in sample])

    all_relevant_documents = get_relevant_documents(supervised_tst)

    precision = calculate_precision(all_relevant_documents, all_recommended_documents)

    return precision, elapsed


model = bolt.UniversalDeepTransformer.load(
    "/Users/nmeisburger/ThirdAI/Universe/msmarco_0_reindexes.bolt"
)
model._get_model().summary()

# test_file = "../msmarco-v2/msmarco/sampled_tst_supervised.csv"
test_file = "/Users/nmeisburger/ThirdAI/data/msmarco/sampled_tst_supervised.csv"

# precision, duration = evaluate_model(model, test_file, use_sparse_inference=False)
# print("Dense precision: ", precision, " Time: ", duration)

model._get_model().ops()[-1].switch_to_hnsw(
    max_nbrs=64, construction_buffer_size=128, search_buffer_size=400
)

model._get_model().summary()

precision, duration = evaluate_model(model, test_file, use_sparse_inference=True)
print("Sparse HNSW precision: ", precision, " Time: ", duration)
