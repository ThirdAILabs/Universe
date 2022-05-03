ARG REV_TAG
FROM thirdai_slim:${REV_TAG}

# Setup shell and current user and env
SHELL ["/bin/bash", "-c"]

# Go into root 
USER root

# Install necessary additional packages for jupyterlab and jupyterlab
RUN apt-get update --yes ; \
    apt-get upgrade --yes ; \
    apt-get install --yes --no-install-recommends \
      gcc \
      g++ \
      make \
      libc6-dev \
      python3-dev ;
RUN pip3 install jupyterlab

# Expose port for running jupyter notebooks
EXPOSE 8888

# Set default starting script, which starts up a jupyter lab server
CMD jupyter lab --ip=0.0.0.0

# Switch back to thirdai user
USER thirdai
