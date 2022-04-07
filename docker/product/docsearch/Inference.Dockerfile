ARG REV_TAG
FROM thirdai_jupyter_release:${REV_TAG}
LABEL Description="Document Search"
SHELL ["/bin/bash", "-c"]

USER root
RUN apt-get -y update ; apt-get -y install git ;

USER thirdai
ADD ColBERT saved 
RUN \
  pip3 install torch tqdm ujson GitPython transformers faiss-cpu pandas ipywidgets; \  
# Install ColBERT wrapper
  cd saved ; \
  pip3 install . 
