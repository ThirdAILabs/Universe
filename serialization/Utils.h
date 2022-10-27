#pragma once

#include <cereal/archives/binary.hpp>
#include <dataset/src/utils/SafeFileIO.h>
#include <fstream>
#include <memory>
#include <string>

namespace thirdai::serialization {

template <typename T>
static inline void saveToFile(T& to_be_serialized,
                              const std::string& filename) {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(filestream);
  oarchive(to_be_serialized);
}

template <typename T>
static inline std::unique_ptr<T> loadFromFile(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  cereal::BinaryInputArchive iarchive(filestream);
  std::unique_ptr<T> deserialize_into(new T());
  iarchive(*deserialize_into);

  return deserialize_into;
}

}  // namespace thirdai::serialization