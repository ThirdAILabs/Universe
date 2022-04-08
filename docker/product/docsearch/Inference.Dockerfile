ARG REV_TAG
FROM thirdai_slim_release:${REV_TAG}
LABEL Description="Document Search Inference"

USER thirdai
ADD ColBERT saved
COPY docsearch_flask_app.py .
RUN \
  # Install for the webserver
  pip3 install flask gunicorn; \
  # Install ColBERT model and dependencies
  pip3 install torch transformers ujson; \  
  cd saved ; \
  pip3 install . 

# Set default starting script, which runs a flask server that serves the
# maxflash index mounted to /home/thirdai/index on the port 5000
ENV FLASK_APP docsearch_flask_app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "-w", "1", "docsearch_flask_app"]
