# Product Dockerfile Instructions

<!-- TODO (Josh): We should probably eventually migrate to Docker compose -->

## What are all of these Dockerfiles?

There are a number of Docker images in this folder. The slim folder contains
the smallest, and is basically just an Ubuntu image with python, pip, and the 
thirdai.so file. The base_jupyter extends the slim image with a jupyer lab 
server. Other folders contain images that extend one of these two images. 
These are our product images.

## How do I build/test/save a Dockerfile?

To build one of the Dockerfiles, run the build_image.sh script in the respective
sub directory. If you want to build for a different platform than the one you 
are on, you may pass in that platform as the first positional argument (this
should be one of linux/arm64 or linux/amd64), which will forward it to the 
Docker build command. 

Note that the github actions integration test currently runs pytest in the base 
directory of the image (the home directory), so put any python tests you want to
run in a subdirectory and call them test_<test name>.py or <test name>_test.py, 
like normal pytest tests.

To build all product Dockerfiles, run
```bash
./build_all_product_dockers.sh <optional platform arg>
```
To test all product Dockerfiles to this directory (this also builds them), run
 ```bash
./test_all_product_dockers.sh <optional platform arg>
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
will also print the full name (including the tag) of the images it is building. 
Note that the git archive that creates the base image only works off of the last 
complete commit, so if you want to test changes you need to create a new commit 
and rerun the build command.

## How do I run a Dockerfiles?

Once you have built a product Docker image, you have two different 
options on how to run it. 

Note that the docker container will start up with the 
thirdai user, which has sudo permission. You may also wish to run the container 
with --privileged, which speeds up the container to native levels (only do this 
if you trust the code in the Docker container; here it is just ours so we trust 
it).

Your options are:
1. Run a jupyter lab server by running 
```bash
docker run -p 8888:8888 <image name>:<git tag> 
```
This will start a jupyter lab server running in the container and forward the 
jupyter lab port out of the container to your local machine. You can then
open the link that gets printed out in your browser and start using jupyter.
2. Use an interactive bash terminal by running
```bash
docker run -it -p 8888:8888 <image name>:<git tag> bash
```
You will enter as the thirdai user which has sudo privledges, and can do 
anything you want, including starting up a jupter lab server.


For example, a typical customer may run
```bash
nohup docker run -p 8888:8888 -v docsearch-work:/home/thirdai/work thirdai_docsearch_release:<git-tag> &
```
which will start a Jupyter Lab server in the background and a new Docker volume
called docsearch-work mounted in /home/thirdai/work. Any changed to the docker
image you make not in a volume won't be saved. You can then cat nohup.out
to get the Jupyter Lab token and url, and then paste it into your browser. If
you are running the container on a remote server, you will need to perform
port forwarding with a command like 
```bash
ssh -v -N -L 8888:localhost:8888 <you>@<remote_server>.
```