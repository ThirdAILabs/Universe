# TODO(josh): Write integration tests for this once we have the github actions
# runner working

ARG REV_TAG
FROM thirdai_slim:${REV_TAG}
LABEL Description="Document Search Inference"

USER thirdai
ADD ColBERT saved
COPY docsearch_flask_app.py .
RUN \
  # Install for the webserver
  pip3 install flask gunicorn; \
  # Install ColBERT model and dependencies. Torch is installed as cpu only
  pip3 install torch transformers ujson  --extra-index-url https://download.pytorch.org/whl/cpu; \  
  cd saved ; \
  pip3 install . 

# Set default starting script, which runs a flask server that serves the
# maxflash index mounted to /home/thirdai/index on the port 5000. Set timeout
# to 600 seconds to allow load time of 10 minutes to load index on startup
ENV FLASK_APP docsearch_flask_app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "-w", "1", "--timeout", "500",  "--log-level", "debug", "docsearch_flask_app:app"]
