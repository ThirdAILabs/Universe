#pragma once

#include <filesystem>
#include <utility>

namespace thirdai::search {

template <typename T>
struct HighestScore {
  using Item = std::pair<T, float>;
  bool operator()(const Item& a, const Item& b) const {
    return a.second > b.second;
  }
};

inline void createDirectory(const std::string& path) {
  if (!std::filesystem::exists(path)) {
    std::filesystem::create_directories(path);
  } else if (!std::filesystem::is_directory(path)) {
    throw std::invalid_argument(
        "Invalid save_path='" + path +
        "'. It must be a directory or not exist and cannot be a file.");
  }
}

}  // namespace thirdai::search