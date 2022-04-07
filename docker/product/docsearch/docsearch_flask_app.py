import numpy as np
import pandas as pd
import time

import thirdai
from thirdai.search import DocRetrieval

from embeddings import DocSearchModel

embedding_model = DocSearchModel()
index_to_query = DocRetrieval.deserialize_from_file("/home/thirdai/index")

from flask import Flask
app = Flask(__name__)

@app.route('/documents', methods=["GET"])
def perform_query_top_1():

    if "query" not in request.args:
        abort(400)
    query_text = request.args.get("query")

    query_embedding = embedding_model.encodeQuery(query_text)

    # For now k is hardcoded to be 8192 for the MaxFlash serarch, and we just 
    # return the top 1 
    internal_top_k = 8192

    result = index_to_query.query(embedding, top_k=internal_top_k)

    return result[0][1]