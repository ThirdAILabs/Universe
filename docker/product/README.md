# Product Dockerfile Instructions

To build a product Dockerfile, run the build_image.sh script in the respective
sub directory. All product Dockerfiles are based on the Dockerfile in the base
folder. Note that the integration test currently runs pytest in the base 
directory of the image (the home directory), so put any python tests you want to
run in a subdirectory and call them test_<test name>.py, like normal pytest tests.

To build all product Dockerfiles, run
```bash
./build_all_product_dockers.sh
```
To save and zip all product Dockerfiles to this directory (this also builds them), run
 ```bash
./save_all_product_dockers.sh
```
To test all product Dockerfiles to this directory (this also builds them), run
 ```bash
./test_all_product_dockers.sh
```

Right now our images are all tagged with the git commit id they were build with
using the result of the command
 ```bash
git log -1 --pretty=format:%h
```
Thus if you make a new commit and run one of the build scripts, you will build
new docker images. This also means you should always specify a tag when running
one of the docker images. To get this tag, you can either list all Docker images
by running  ```bash docker images ``` and choose the most recent one, or by
yourself running  ```bash git log -1 --pretty=format:%h```. The build script
will also print the images it is building.

Once you have built/distributed a product Docker image, you have different 
options on how to run it.
You can run any of the commands on https://jupyter-docker-stacks.readthedocs.io/en/latest/
to start a jupyter notebook. Note that you may need to SSH tunnel to view the
Jupyter labs gui locally if you are running the container on a remote machine. 
Also note that the Jupyter lab startup script 
(https://github.com/jupyter/docker-stacks/blob/master/base-notebook/start.sh)
does lots of nice things, like by default running with jovyan as a user (where
we have preistalled pip packes), allowing you to grant your user sudo with
GRANT_SUDO=yes, and allowing you to specify any user and then copying/linking
the jovyan home directory.
You can also run
```bash
docker run --user 1000:100 docsearch:<git tag> -it bash 
```
to just jump into bash as the jovyan user, and
```bash
docker run:<git tag> --user 1000:100 docsearch pytest
```
to run basic integration tests as the jovyan user.

For example, you may run
```bash
nohup docker run --rm -p 10000:8888 -v docsearch-work:/home/jovyan/work docsearch:<git-tag> &
```
which will start a Jupyter Lab server in the background and a new Docker volume
called docsearch-work mounted in /home/jovyan/work. You can cat nohup.out
to get the Jupyter Lab token and url, and then paste it into your browser
(changing the port to 10000), possibly performing port forwarding with a 
command like 
```bash
ssh -v -N -L 10000:localhost:10000 <you>@<remote_server>.
```

Note that you can run the jupyter notebook containers with GRANT_SUDO=yes
to give the default jovyan user sudo permissions. You may also wish to run 
the container with --privileged, which speeds up the container to native levels 
(only do this if you trust the code in the Docker container; here it is just
ours).

<!-- TODO (Josh): We should probably eventually migrate to Docker compose -->