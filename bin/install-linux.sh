# Run this script as sudo
apt update

# Install cmake
apt install cmake -y 

# Install clang-format, clang-tidy
apt install clang-format -y
apt install clang-tidy -y

# Append Universe build directory to PYTHONPATH on interactive shell startup
echo "export PYTHONPATH=~/Universe/build:$PYTHONPATH" >> $HOME/.bash_profile 

# Install necessary python packages
pip3 install dark
pip3 install pytest