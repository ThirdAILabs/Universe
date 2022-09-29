import numpy as np
import time

import thirdai
from thirdai.search import DocRetrieval

from thirdai.embeddings import DocSearchModel

import torch

from flask import Flask, request, abort, jsonify
import sys

app = Flask(__name__)

# Set logger to be gunicorn
import logging

gunicorn_logger = logging.getLogger("gunicorn.error")
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

app.logger.info("Loading index and model")
embedding_model = DocSearchModel("/home/thirdai/saved")
centroids = torch.from_numpy(embedding_model.getCentroids())
index_to_query = DocRetrieval.deserialize_from_file("/home/thirdai/index")
app.logger.info("Index and model loaded")


@app.route("/documents", methods=["GET"])
def perform_query_top_1():

    if "query" not in request.args:
        app.logger.error(f"query not in args, args were {request.args}, aborting")
        abort(400)

    if "top_k" not in request.args:
        app.logger.error(f"top_k not in args, args were {request.args}, aborting")
        abort(400)

    query_text = request.args.get("query")

    top_k = int(request.args.get("top_k"))

    app.logger.debug(
        f'Handling document search query: query="{query_text}", top_k="{top_k}"'
    )

    start = time.time()

    query_embedding = embedding_model.encodeQuery(query_text)

    multiplied = centroids @ query_embedding.T

    centroid_ids = torch.argmax(multiplied, dim=0)

    result = index_to_query.query(
        query_embedding.numpy(), centroid_ids.numpy(), top_k=top_k
    )

    app.logger.info(
        f'For query="{query_text}" and top_k="{top_k}", found {len(result)} result(s) in {time.time() - start} seconds'
    )

    return jsonify(doc_ids=[r[0] for r in result], doc_texts=[r[1] for r in result])
