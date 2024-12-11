#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <rocksdb/slice.h>
#include <sstream>

namespace thirdai::search::ndb {

template <typename T>
std::string serialize(const T& data) {
  std::stringstream out;

  cereal::BinaryOutputArchive ar(out);
  ar(data);

  return out.str();
}

// This struct is used to wrap a char* into a stream, see
// https://stackoverflow.com/questions/7781898/get-an-istream-from-a-char
struct Membuf : std::streambuf {
  Membuf(char* begin, char* end) { this->setg(begin, begin, end); }
};

template <typename T>
T deserialize(rocksdb::Slice& bytes) {
  char* ptr = const_cast<char*>(bytes.data());
  Membuf buf(ptr, ptr + bytes.size());

  std::istream in(&buf);
  cereal::BinaryInputArchive ar(in);

  T data;
  ar(data);

  return data;
}

}  // namespace thirdai::search::ndb