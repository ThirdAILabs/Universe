#pragma once

#include <cereal/archives/binary.hpp>
#include <dataset/src/utils/SafeFileIO.h>
#include <fstream>
#include <memory>
#include <string>

namespace thirdai::serialization {

template <typename T>
static inline void simpleSaveToFile(T& to_be_serialized,
                                    const std::string& filename) {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(filestream);
  oarchive(to_be_serialized);
}

template <typename T>
static inline std::unique_ptr<T> simpleLoadFromFile(
    T* empty_object, const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  cereal::BinaryInputArchive iarchive(filestream);
  std::unique_ptr<T> deserialize_into(empty_object);
  iarchive(*deserialize_into);

  return deserialize_into;
}

#define ADD_SIMPLE_SAVE_LOAD_METHODS(T)                          \
  void save(const std::string& filename) {                       \
    serialization::simpleSaveToFile(*this, filename);            \
  }                                                              \
                                                                 \
  static std::unique_ptr<T> load(const std::string& filename) {  \
    return serialization::simpleLoadFromFile(new T(), filename); \
  }

}  // namespace thirdai::serialization