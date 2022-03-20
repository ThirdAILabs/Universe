# DocSearch Dockerfile

To use this Dockerfile, cd to this directory, then run
```bash
./create_docsearch_dir.sh
```
This will zip the current state of Universe into this directory, so it can be copied into the
container (for official versions to customers, only run this on main). Next, run
 ```bash
docker build -t docsearch .
```
to build the docker container. You then have different options to run it.
You can run any of the commands on https://jupyter-docker-stacks.readthedocs.io/en/latest/
to start a jupyter notebook. You can also run
```bash
docker run docsearch -it bash 
```
to run bash, and
```bash
docker run docsearch pytest
```
to run basic integration tests.