
export CIBW_ENVIRONMENT_LINUX="
CMAKE_ARGS=\"-DOPENSSL_ROOT_DIR:PATH=/usr/bin/openssl3\" 
THIRDAI_BUILD_IDENTIFIER=$(git rev-parse --short HEAD) OPENAI_API_KEY=$OPENAI_API_KEY"

export CIBW_BEFORE_ALL="
    sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo && \
    sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo && \
    sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo && \
    yum update -y && \
    yum clean all && \
    yum install -y gcc make perl-core zlib-devel wget pcre-devel && \
    wget https://www.openssl.org/source/openssl-3.0.13.tar.gz && \
    tar -xzvf openssl-3.0.13.tar.gz && \
    cd openssl-3.0.13 && \
    ./config --prefix=/usr --openssldir=/etc/ssl --libdir=lib no-shared zlib-dynamic && \
    make && \
    make test && \
    make install && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64' > /etc/profile.d/openssl.sh && \
    source /etc/profile.d/openssl.sh && \
    openssl version -a && \
    yum install -y libxslt-devel libxml2-devel
"

export CIBW_BUILD="cp39-manylinux_x86_64"
export CIBW_BUILD_VERBOSITY=3
export CIBW_SKIP="cp36-* cp37-* cp38-* cp310-* cp311-*"
cibuildwheel --output-dir wheelhouse --platform linux