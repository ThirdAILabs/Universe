import numpy as np
import time

import thirdai
from thirdai.search import DocRetrieval

from embeddings import DocSearchModel


from flask import Flask, request, abort, jsonify
import sys

app = Flask(__name__)

# Set logger to be gunicorn
import logging

gunicorn_logger = logging.getLogger("gunicorn.error")
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(logging.INFO)

app.logger.info("Loading index and model")
embedding_model = DocSearchModel()
index_to_query = DocRetrieval.deserialize_from_file("/home/thirdai/index")
app.logger.info("Index and model loaded")


@app.route("/documents", methods=["GET"])
def perform_query_top_1():

    if "query" not in request.args:
        app.logger.info(f"query not in args, args were {request.args}, aborting")
        abort(400)

    if "top_k" not in request.args:
        app.logger.info(f"top_k not in args, args were {request.args}, aborting")
        abort(400)

    query_text = request.args.get("query")

    top_k = int(request.args.get("top_k"))

    app.logger.info(
        f'Handling document search query: query="{query_text}", top_k="{top_k}"'
    )

    query_embedding = embedding_model.encodeQuery(query_text)

    # For now k is hardcoded to be 8192 for the MaxFlash serarch, and we just
    # return the top 1
    internal_top_k = 8192

    result = index_to_query.query(query_embedding, top_k=internal_top_k)

    app.logger.info(
        f'For query="{query_text}", found {len(result)} results, returning {min(len(result), top_k)} results'
    )

    return jsonify(
        doc_ids=[r[0] for r in result[:top_k]], doc_texts=[r[1] for r in result[:top_k]]
    )
