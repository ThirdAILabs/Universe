# Use the CentOS base image
FROM centos:7

RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo && \
sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo && \
sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo 

# Install necessary packages
# RUN yum update -y && \
#     yum install -y python39 python39-pip python39-devel && \
#     yum install -y gcc gcc-c++ make

RUN yum update -y && yum clean all
RUN yum -y install epel-release && yum clean all
RUN yum -y install python39-pip && yum clean all

RUN yum install -y gcc gcc-c++ make

# RUN yum install openssl11-devel -y && \
# yum install -y libxslt-devel libxml2-devel

RUN yum install -y make gcc perl-core pcre-devel wget zlib-devel
RUN wget https://ftp.openssl.org/source/openssl-1.1.1k.tar.gz && tar -xzvf openssl-1.1.1k.tar.gz && cd openssl-1.1.1k && ./config --prefix=/usr --openssldir=/etc/ssl --libdir=lib no-shared zlib-dynamic && make && make test && make install && 

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# CMD ["sh", "-c", "opensslpython3.9 -c \"import ssl\" && python3.9 setup.py bdist_wheel && mkdir -p /output && cp dist/*.whl /output/ && cp /output/*.whl /app/ && ls -l /app"]
CMD ["/bin/bash"]