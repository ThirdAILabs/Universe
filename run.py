from thirdai import neural_db as ndb, licensing
licensing.activate("D0F869-B61466-6A28F0-14B8C6-0AC6C6-V3")

import glob
import os
import pandas as pd

bazaar_dir = "./bazaar_cache"
if not os.path.isdir(bazaar_dir):
    os.mkdir(bazaar_dir)

from pathlib import Path
from thirdai.neural_db import Bazaar
bazaar = Bazaar(cache_dir=Path(bazaar_dir))

bazaar.fetch()
print(bazaar.list_model_names())

db = bazaar.get_model("General QnA")


insertable_docs = []

csv_files = glob.glob("/home/thirdai/Demos/neural_db/ntt/ntt-data/all_data/*.csv")

for file in csv_files:
    df = pd.read_csv(file, nrows=0)
    col_names = list(df.keys())
    csv_doc = ndb.CSV(
        path=file,
        strong_columns=[],
        weak_columns=col_names,  
        reference_columns=col_names,
        index_columns=col_names)
    #
    insertable_docs.append(csv_doc)

print("inserting into ndb", flush=True)
source_ids = db.insert(insertable_docs, train=False)

