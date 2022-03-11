FROM jupyter/base-notebook:python-3.9.10
LABEL Description="Build ThirdAI"
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
    cmake \
    gdb ; \
    cd /Universe ; \
    bin/build.sh ;


FROM jupyter/base-notebook:python-3.9.10
LABEL Description="Document Search"
SHELL ["/bin/bash", "-c"]
USER root
RUN pip3 install torch pytest numpy
COPY --from=0 /Universe/build/thirdai.cpython-* ./
# This line is temporary, eventually we will test with a custom script
COPY --from=0 /Universe/flash/python_tests/*doc_retrieval* ./
RUN echo "export PYTHONPATH=~" >> ~/.bashrc
