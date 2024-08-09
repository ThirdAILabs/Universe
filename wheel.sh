export CIBW_ENVIRONMENT_LINUX="CMAKE_ARGS=\"-DOPENSSL_ROOT_DIR:PATH=/usr/lib64/openssl11;/usr/include/openssl11\" THIRDAI_BUILD_IDENTIFIER=$(git rev-parse --short HEAD) OPENAI_API_KEY=$OPENAI_API_KEY OPENSSL_FORCE_FIPS_MODE=1"
# export CIBW_BEFORE_BUILD_LINUX="sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo && sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo && sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo && yum update -y && yum clean all && yum install openssl11-devel -y && yum install -y libxslt-devel libxml2-devel"
export CIBW_BEFORE_ALL="
    sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo && \
    sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo && \
    sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo && \
    yum update -y && \
    yum clean all && \
    yum install openssl11 openssl11-devel -y && \
    openssl version -a && \
    which openssl && \
    which openssl11 && \
    openssl11 version -a && \
    mv /usr/bin/openssl /usr/bin/openssl_backup && \
    ln -s /usr/bin/openssl11 /usr/bin/openssl && \
    openssl version -a && \
    yum install -y libxslt-devel libxml2-devel"

export CIBW_BUILD="cp39-manylinux_x86_64"
export CIBW_BUILD_VERBOSITY=3
export CIBW_SKIP="cp36-* cp37-* cp38-* cp310-* cp311-*"

cibuildwheel --output-dir wheelhouse --platform linux



