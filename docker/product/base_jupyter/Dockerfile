# See https://github.com/moby/moby/issues/34482 for why we need this stage
ARG REV_TAG
FROM thirdai_slim_release:${REV_TAG} AS stageforcopy

FROM jupyter/base-notebook:python-3.8.8
SHELL ["/bin/bash", "-c"]
USER root
# Copy Universe so file
COPY --from=stageforcopy thirdai.cpython-* ./
# Copy python integration tests
COPY --from=stageforcopy /*.py ./
RUN pip3 install numpy pytest ; 
USER jovyan
RUN \
    echo "export PYTHONPATH=~" >> ~/.bashrc ; \
    ipython profile create ; \
    echo "c.InteractiveShellApp.exec_lines = ['import sys; sys.path.insert(0, \'/home/jovyan\'); sys.path.insert(0, \'/home/jovyan/msmarco/colbert_modified\')']" \
      >> ~/.ipython/profile_default/ipython_config.py