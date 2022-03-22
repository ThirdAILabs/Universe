FROM jupyter/base-notebook:python-3.9.10
SHELL ["/bin/bash", "-c"]
ADD Universe.tar.bz2 /Universe
USER root
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --yes ; \
    apt-get upgrade --yes ; \
    apt-get install --yes --no-install-recommends \
    build-essential \
    git \
    curl \
    cmake
RUN /Universe/bin/build.sh


FROM jupyter/base-notebook:python-3.9.10
SHELL ["/bin/bash", "-c"]
USER root
COPY --from=0 /Universe/build/thirdai.cpython-* ./
COPY thirdai_smoke_test.py thirdai_smoke_test.py
RUN pip3 install numpy pytest ; 
USER jovyan
RUN \
    echo "export PYTHONPATH=~" >> ~/.bashrc ; \
    ipython profile create ; \
    echo "c.InteractiveShellApp.exec_lines = ['import sys; sys.path.insert(0, \'/home/jovyan\'); sys.path.insert(0, \'/home/jovyan/msmarco/colbert_modified\')']" \
      >> ~/.ipython/profile_default/ipython_config.py