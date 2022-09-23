#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace thirdai::compression::serializer {

/*
 * These are helper classes to serialize and deserialize c++ objects.
 *
 * InputHelper takes in a char array that already has serialized data stored in
 * it and helps in reading data from it.
 *
 * OutputHelper takes in a char array(let's call it serialized_data) with memory
 * already allocated to it and writes data onto this memory. Your object should
 * return serialized_data as the memory pointed to by serialized_data is the
 * place where helper writes the object data.
 *
 * Note: These are just helper classes and do not "own" the data as such and
 * meant to be used locally by a function.
 *
 * Why not cereal?
 * From the docs: "cereal archives operate on either an std::ostream or
 * std::istream object". After writing data on a stream, we need to transfer the
 * data to python without copying. This is hard to implement with streams.
 * TODO(Shubh): Transfer cereal data to python without copying.
 */
class BinaryInputHelper {
 public:
  explicit BinaryInputHelper(const char* data) : _ptr(data) {}

  template <class T>
  void read(T* data) {
    std::memcpy(data, _ptr, sizeof(T));
    _ptr += sizeof(T);
  }

  template <class T>
  void readVector(std::vector<T>& vector) {
    uint32_t size;
    read(&size);

    vector.resize(size);
    std::memcpy(vector.data(), _ptr, sizeof(T) * size);
    _ptr += sizeof(T) * size;
  }

 private:
  const char* _ptr;
};

class BinaryOutputHelper {
 public:
  explicit BinaryOutputHelper(char* data) : _ptr(data) {}

  template <class T>
  void write(const T* data) {
    std::memcpy(_ptr, data, sizeof(T));
    _ptr += sizeof(T);
  }

  template <class T>
  void writeVector(const std::vector<T>& vector) {
    uint32_t size = vector.size();
    write(&size);

    std::memcpy(_ptr, vector.data(), sizeof(T) * size);
    _ptr += sizeof(T) * size;
  }

 private:
  char* _ptr;
};
}  // namespace thirdai::compression::serializer