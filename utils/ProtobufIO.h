#pragma once

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <ostream>

namespace thirdai::utils {

using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;

class ProtobufWriter {
 public:
  explicit ProtobufWriter(std::shared_ptr<ZeroCopyOutputStream> output);

  void serialize(const google::protobuf::MessageLite& object);

 private:
  std::shared_ptr<ZeroCopyOutputStream> _output;
};

class ProtobufReader {
 public:
  explicit ProtobufReader(std::shared_ptr<ZeroCopyInputStream> input);

  void deserialize(google::protobuf::MessageLite& object);

 private:
  std::shared_ptr<ZeroCopyInputStream> _input;
};

}  // namespace thirdai::utils