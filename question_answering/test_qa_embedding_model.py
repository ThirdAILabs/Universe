from tensorflow import keras
from transformers import BertTokenizer
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

with open("/share/josh/msmarco/queries.dev.small.tsv") as f:
    qid_map = [int(line.split()[0]) for line in f.readlines()]

with open("/share/josh/msmarco/queries.dev.small.tsv") as f:
    queries = [line.split("\t")[1].strip() for line in f.readlines()]
    tokenized_queries = []
    for q in tqdm(queries):
        # Remove start and end tokens
        tokenized_queries.append(tokenizer(q)["input_ids"][1:-1])


with open("/share/josh/msmarco/collection.tsv") as f:
    documents = [line.split("\t")[1].strip() for line in f.readlines()]
    tokenized_documents = []
    for q in tqdm(documents):
        # Remove start and end tokens
        tokenized_documents.append(tokenizer(q)["input_ids"][1:-1])
