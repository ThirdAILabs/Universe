from thirdai import neural_db as ndb
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument("unsup")
parser.add_argument("test")
parser.add_argument("-e", "--epochs", default=[10], type=int, nargs="+")

args = parser.parse_args()

for epochs in args.epochs:
    print(epochs, type(epochs))

df = pd.read_csv(args.test)
query_batches = [df["QUERY"][start:start + 2048] for start in range(0, len(df), 2048)]
docid_batches = [df["DOC_ID"][start:start + 2048] for start in range(0, len(df), 2048)]

db = ndb.NeuralDB(use_inverted_index=False)
doc = ndb.CSV(args.unsup, id_column="DOC_ID", strong_columns=["TITLE"], weak_columns=["TEXT"])

first = True
for epochs in args.epochs:
    if not first:
        print("Clearing...")
        db.clear_sources()
    for _ in range(epochs):
        db.insert([doc], epochs=1)

        correct = 0
        for query_batch, docid_batch in zip(query_batches, docid_batches):
            answers = db.search_batch(query_batch, top_k=1)
            correct += sum([answers[i][0].id == docid for i, docid in enumerate(docid_batch)])
        print("P@1:", correct / len(df))
    first = False


