#!/bin/bash

BASEDIR=$(dirname "$0")
BUILDDIR="$BASEDIR/../build"

# Download and unzip data
SVMDATADIR="$BUILDDIR/utils/tests/dataset/svm" 
echo $SVMDATADIR
curl "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/bibtex.bz2" --output $SVMDATADIR/bibtex.bz2
bzip2 -d $SVMDATADIR/bibtex.bz2
STRINGDATADIR="$BUILDDIR/utils/tests/dataset/string"
wget -O "$STRINGDATADIR/FreelandSep10_2020.txt" "https://storage.googleapis.com/kagglesdsdata/datasets/891801/1516980/FreelandSep10_2020.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210927%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210927T215126Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=1b09798ca355edcc68c6d19ed047c7d90775ca27976bf796e72579976908b306b7e6a543fb52770baf0686e1040b5724f1ced4d8d6a582f7d1725cba4534d19d698ad4759b56ca454ecb88bbdeb90320cfc74177565eca8ca46331a1817724582b9bb41d4b30310d601c4cb0331f9c75f36dd977503fe6160d66bdd271366ff317b30ecf913aec73d6cff32e894bc1d699a242cfaea81209c35a1f79abfdea2aa009fdc98ffca146a429a99501203657e4868cd1f88a9c22b9643e8a4b7359f3bf6311340323ac14499fde298ccf9c4251a436e432cf9de74cbc8a5261205acfbf3d9a51aa7a122136cb484bb46bbdaf17347e5235b2b8e2cef678c5fc0a2f8c"
