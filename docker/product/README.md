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

Once you have built/distributed a product Docker image, you have different 
options on how to run it.
You can run any of the commands on https://jupyter-docker-stacks.readthedocs.io/en/latest/
to start a jupyter notebook. Note that you may need to SSH tunnel if you are 
running the container on a remote machine. 
You can also run
```bash
docker run docsearch -it bash 
```
to just jump into bash, and
```bash
docker run docsearch pytest
```
to run basic integration tests.

For example, you may run
```bash
nohup docker run --rm -p 10000:8888 -v "${PWD}":/home/jovyan/work docsearch &
```
which will start a Jupyter Lab server in the background. You can read nohup.out
to get the Jupyter Lab token and url, and then paste it into your browser
(changing the port to 10000), possibly performing port forwarding with a 
command like 
```bash
ssh -v -N -L 10000:localhost:10000 <you>@<remote_server>.
```

Note that you can run the jupyter notebooks or the bash terminal with GRANT_SUDO=yes
to give the default jovyan user sudo permissions. 