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
brew install gcc@11
ln -s /opt/homebrew/bin/gcc-11 /opt/homebrew/bin/gcc
ln -s /opt/homebrew/bin/g++-11 /opt/homebrew/bin/g++

# Install cmake
brew install cmake

# Git line endings
git config --global core.eol lf 
git config --global core.autocrlf input