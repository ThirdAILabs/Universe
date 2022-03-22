FROM base_product
LABEL Description="Document Search"
SHELL ["/bin/bash", "-c"]
USER root
RUN \
  apt-get -y update ; \
  apt-get -y install git ; \
  pip3 install torch pytest numpy tqdm ujson GitPython transformers faiss-cpu pandas ipywidgets; 