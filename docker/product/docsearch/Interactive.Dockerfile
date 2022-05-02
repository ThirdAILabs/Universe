ARG REV_TAG
FROM thirdai_jupyter_interactive:${REV_TAG}
LABEL Description="Document Search"
SHELL ["/bin/bash", "-c"]

USER root
RUN apt-get -y update ; apt-get -y install git ;

USER thirdai
ADD ColBERT saved 
RUN \
  # Install ColBERT model and dependencies. Torch is installed as cpu only.
  pip3 install torch transformers ujson --extra-index-url https://download.pytorch.org/whl/cpu; \  
  cd saved ; \
  pip3 install .
