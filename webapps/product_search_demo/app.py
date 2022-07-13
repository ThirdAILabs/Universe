import time
from flask import Flask, request, render_template
import sys
import warnings

warnings.filterwarnings("ignore")

import thirdai
from thirdai.search import DocRetrieval
from thirdai.embeddings import DocSearchModel

product_index = None
embedding_model = DocSearchModel()

app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "home.html",
    )


@app.route("/", methods=["POST"])
def query_products():
    query_text = request.form["query"]

    start = time.time()
    embedding = embedding_model.encodeQuery(query_text)
    results = product_index.query(embedding, top_k=4)
    total_time = round(time.time() - start, 3)

    products = [x[1] for x in results]

    query_time = f"Query completed in {total_time} seconds."
    return render_template("home.html", products=products, time=query_time)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Expected path to saved index as command line argument.")

    product_index = DocRetrieval.deserialize_from_file(sys.argv[1])

    # Set host = 0.0.0.0 so that the app is accessible outside of local via the machines ip address.
    app.run(debug=False, host="0.0.0.0")
