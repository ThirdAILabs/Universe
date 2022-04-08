ARG REV_TAG
FROM thirdai_jupyter_release:${REV_TAG}
LABEL Description="Document Search"
SHELL ["/bin/bash", "-c"]

USER root
RUN apt-get -y update ; apt-get -y install git ;

USER thirdai
ADD ColBERT saved 
RUN \
  # Install ColBERT model and dependencies
  pip3 install torch transformers ujson; \  
  cd saved ; \
  pip3 install .