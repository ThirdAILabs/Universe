ARG REV_TAG
FROM thirdai_jupyter_release:${REV_TAG}
LABEL Description="Text Classification"
SHELL ["/bin/bash", "-c"]
USER root
RUN \
  apt-get -y update ; \
  apt-get -y install git ; \
  pip3 install torch tqdm ujson GitPython transformers pandas numpy pytest; 
USER thirdai