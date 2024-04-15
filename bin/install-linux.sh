# Run this script as sudo
apt update

# Install cmake
apt install cmake -y 

# Install clang-format, clang-tidy
apt install clang-format -y
apt install clang-tidy -y

# Install libcurl dependencies
apt-get install curl libssl-dev libcurl4-openssl-dev

# Install necessary python packages
pip3 install dark
pip3 install pytest
pip3 install mlflow
pip3 install toml
pip3 install psutil