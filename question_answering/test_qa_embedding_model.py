from tensorflow import keras
from transformers import BertTokenizer
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm
from embedding_model import get_compiled_triplet_model

triplet_network, sentence_embedding_model = get_compiled_triplet_model(
    learning_rate=0.01
)
triplet_network.load_weights(filepath=".mdl_wts.hdf5")

result_file_name = "temp_ranking.txt"

tokenized_queries = np.load("tokenized_queries_test.npy")
embedded_queries = sentence_embedding_model.predict(tokenized_queries)
embedded_queries = tf.math.l2_normalize(embedded_queries, axis=1).numpy()

tokenized_documents = np.load("tokenized_documents_test.npy")
all_topks = []
num_batches = 10
doc_offsets = [len(tokenized_documents) // num_batches * i for i in range(num_batches)]
doc_offsets.append(len(tokenized_documents))
embedded_documents = []
for batch_start, batch_end in zip(doc_offsets[:-1], doc_offsets[1:]):
    embedded_documents.append(
        tf.math.l2_normalize(
            sentence_embedding_model.predict(
                tokenized_documents[batch_start:batch_end]
            ),
            axis=1,
        ).numpy()
    )
embedded_documents = np.concatenate(embedded_documents)

all_topks = []
for i in tqdm(range(len(embedded_queries))):
    dot = np.dot(embedded_queries[i], embedded_documents.T)
    top_k = tf.math.top_k(dot, k=1000, sorted=True).indices.numpy()
    all_topks.append(top_k)

all_topks = np.array(all_topks)
print(all_topks.shape)
print(all_topks[:3])

from ms_marco_eval import compute_metrics_from_files

with open("/share/josh/msmarco/queries.dev.small.tsv") as f:
    qid_map = [int(line.split()[0]) for line in f.readlines()]

with open(result_file_name, "w") as f:
    for qid_index, r in enumerate(all_topks):
        for rank, pid in enumerate(r):
            qid = qid_map[qid_index]
            f.write(f"{qid}\t{pid}\t{rank + 1}\n")

metrics = compute_metrics_from_files(
    "/share/josh/msmarco/qrels.dev.small.tsv", result_file_name
)
print("#####################")
for metric in sorted(metrics):
    print("{}: {}".format(metric, metrics[metric]))
print("#####################", flush=True)
