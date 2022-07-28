#!/bin/bash

# Run this script with sudo

# Install homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Make homebrew path come first
echo "export PATH=/opt/homebrew/bin:\$PATH" >> $HOME/.bash_profile

# Install clang-tidy, clang-format
brew install llvm
ln -s "$(brew --prefix llvm)/bin/clang-format" "/usr/local/bin/clang-format"
ln -s "$(brew --prefix llvm)/bin/clang-tidy" "/usr/local/bin/clang-tidy"
ln -s "$(brew --prefix llvm)/bin/clang-apply-replacements" "/usr/local/bin/clang-apply-replacements"

# Install gcc
# We want a stable version of gcc 11.2 (needed for cryptopp to compile correctly), 
# so we checkout the latest version of brew that has this version, install gcc,
# pin it, and then return brew to the correct version. The version info of
# gcc or g++ then is:
#               g++ (Homebrew GCC 11.2.0_3) 11.2.0
# See https://stackoverflow.com/questions/39187812/homebrew-how-to-install-older-versions
# If you have already installed a newer version of gcc, you can uninstall it  (brew uninstall gcc) 
# and then run these commands again. The linking below won't need to be rerun.
cd "$(brew --repo homebrew/core)"
git checkout 447f94e742c7b88875a659b6c857d164ef8c6a2c
HOMEBREW_NO_AUTO_UPDATE=1 brew install gcc
git checkout master
brew pin gcc

# We link the homebrew gcc and g++ symbols to gcc-11 and g++-11
# This may not be needed anymore since we go to a point in the homebrew git log
# where we are sure that "gcc" refers to gcc-11, but for now we will keep this 
# in case it is still needed (the next time someone does a clean install we
# can see if it is really neccesary).
ln -s /opt/homebrew/bin/gcc-11 /opt/homebrew/bin/gcc
ln -s /opt/homebrew/bin/g++-11 /opt/homebrew/bin/g++

# Install cmake
brew install cmake

# Git line endings
git config --global core.eol lf 
git config --global core.autocrlf input

# Install necessary python packages
# Note if using an m1 mac you may need to download conda and use it to install 
# toml and mlflow.
pip3 install dark
pip3 install pytest
pip3 install psutil
# pip3 install mlflow
# pip3 install toml