
export CIBW_ENVIRONMENT_LINUX="
CMAKE_ARGS=\"-DOPENSSL_ROOT_DIR:PATH=/usr/local/ssl\" 
THIRDAI_BUILD_IDENTIFIER=$(git rev-parse --short HEAD) 
OPENAI_API_KEY=$OPENAI_API_KEY 
OPENSSL_FORCE_FIPS_MODE=1"

export CIBW_BEFORE_ALL="
    sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo && \
    sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo && \
    sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo && \
    yum update -y && \
    yum clean all && \
    yum install -y wget && \   
    yum install -y gcc make perl-core zlib-devel curl && \
    cd /tmp && \
    wget https://www.openssl.org/source/openssl-3.0.13.tar.gz && \
    tar xzf openssl-3.0.13.tar.gz && \
    cd openssl-3.0.13 && \
    ./config --prefix=/usr/local/ssl --openssldir=/usr/local/ssl shared zlib && \
    make && \
    make install && \
    echo '/usr/local/ssl/lib64' > /etc/ld.so.conf.d/openssl-3.0.13.conf && \
    ldconfig && \
    mv /usr/bin/openssl /usr/bin/openssl_backup && \
    ln -s /usr/local/ssl/bin/openssl /usr/bin/openssl && \
    openssl version -a && \
    yum install -y libxslt-devel libxml2-devel
"

export CIBW_BUILD="cp39-manylinux_x86_64"
export CIBW_BUILD_VERBOSITY=3
export CIBW_SKIP="cp36-* cp37-* cp38-* cp310-* cp311-*"
cibuildwheel --output-dir wheelhouse --platform linux