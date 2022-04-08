# Product Dockerfile Instructions

<!-- TODO (Josh): We should probably eventually migrate to Docker compose -->

## What are all of these Dockerfiles?

There are a number of Docker images in this folder. The slim folder contains
the smallest, and is basically just an Ubuntu image with python, pip, and the 
thirdai.so file. The base_jupyter extends the slim image with a JupyerLab 
server. Other folders contain Inference images that extend the slim image and
Interactive images that extend the jupyter notebook image. 
These are our product images.

## How do I build/test/save a Dockerfile?

To build one of the Dockerfiles, run the build_inference_image.sh or
build_interactive_image.sh script in the respective sub directory. You may also 
need to build the dependencies of that image first. The Interactive Dockerfiles 
depend on the base_jupyter Dockerfile, which depends on the slim Dockerfile. The
Inference Dockerfiles just depend on the slim Dockerfile. 

If you want to build for a different platform than the one you 
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

See thirdai.com/resources/documentation/dockerimages/ (or if this isn't up
yet see https://www.notion.so/Use-ThirdAI-Docker-Container-c3524be84fec486f92122caff6ad4696).

## How do I create a new Dockerfile?

To make a new Interactive Dockerfile, create an Interactive.Dockerfile and a
build_interactive_image.sh script modeled off of one of the existing ones
(e.g. docsearch). Then simply add the build command to build_all_product_dockers.sh
and increment the expected count in build_all_product_dockers.sh by 1. A new
Inference Dockerfile can be created in a very similar way by going off of the
existing examples.