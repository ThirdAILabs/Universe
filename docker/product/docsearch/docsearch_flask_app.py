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
app.logger.setLevel(logging.DEBUG)

app.logger.debug("Loading index and model")
embedding_model = DocSearchModel()
index_to_query = DocRetrieval.deserialize_from_file("/home/thirdai/index")
app.logger.debug("Index and model loaded")


@app.route("/documents", methods=["GET"])
def perform_query_top_1():

    app.logger.debug("Handling document search query")

    if "query" not in request.args:
        app.logger.debug(f"query not in args, args were {request.args}, aborting")
        abort(400)

    if "top_k" not in request.args:
        app.logger.debug(f"top_k not in args, args were {request.args}, aborting")
        abort(400)

    query_text = request.args.get("query")

    top_k = int(request.args.get("top_k"))

    app.logger.debug(f'Query text is "{query_text}"')

    query_embedding = embedding_model.encodeQuery(query_text)

    # For now k is hardcoded to be 8192 for the MaxFlash serarch, and we just
    # return the top 1
    internal_top_k = 8192

    result = index_to_query.query(query_embedding, top_k=internal_top_k)

    return jsonify(results=[r[0] for r in result[:top_k]])
