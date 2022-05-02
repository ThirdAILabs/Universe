
# Building the docs
There are different types of documentation you can build.
## HTML
First, you will have to install the theme we are currently using:
```
pip3 install sphinx-rtd-theme
```
Then run 
```
make html
```
in this folder. The generated html files will be in docs/_build/html, and you
can view them by opening docs/_build/html/index.html in a web browser.
<!-- TODO: Host these html docs on a website. -->
## Markdown
First, you will have to install the sphinx markdown builder:
```
pip3 install sphinx-markdown-builder
```
Then run
```
make markdown
```
and the markdown files will be in docs/_build/markdown. 
<!-- TODO: Copy the markdown files to our demo repo to make them public, once the
bolt documentation is done. -->
## Anything else (man, json, etc,)
To get a list of possible target, run 
'''
make
'''
Then choose a target you want to run 
'''
make target
''''
and the results will be in _build/target.
