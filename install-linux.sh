# Run this script as sudo

# Add Bazel distribution URI as a package source
apt install apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list

# Install and update Bazel
apt update -y && apt install bazel -y
apt update -y && apt full-upgrade -y

# Install clang0format, clang-tidy
apt install clang-format -y
apt install clang-tidy -y