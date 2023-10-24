#include "ProtobufIO.h"
#include <google/protobuf/io/coded_stream.h>
#include <stdexcept>

using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;

namespace thirdai::utils {

// See this stackoverflow post for how this works.
// https://stackoverflow.com/questions/2340730/are-there-c-equivalents-for-the-protocol-buffers-delimited-i-o-functions-in-ja

ProtobufWriter::ProtobufWriter(std::shared_ptr<ZeroCopyOutputStream> output)
    : _output(std::move(output)) {}

void ProtobufWriter::serialize(const google::protobuf::MessageLite& object) {
  // We create a new coded stream for each message.  This is not an overhead.
  CodedOutputStream output(_output.get());

  // Write the size of the object.
  size_t size = object.ByteSizeLong();
  output.WriteVarint64(size);

  uint8_t* buffer = output.GetDirectBufferForNBytesAndAdvance(size);
  if (buffer != NULL) {
    // Optimization:  The message fits in one buffer, so use the faster
    // direct-to-array serialization path.
    object.SerializeWithCachedSizesToArray(buffer);
  } else {
    // Slightly-slower path when the message is multiple buffers.
    object.SerializeWithCachedSizes(&output);
    if (output.HadError()) {
      throw std::runtime_error("Error serializing model.");
    }
  }
}

ProtobufReader::ProtobufReader(std::shared_ptr<ZeroCopyInputStream> input)
    : _input(std::move(input)) {}

void ProtobufReader::deserialize(google::protobuf::MessageLite& object) {
  // We create a new coded stream for each message.  This is not an overhead.
  google::protobuf::io::CodedInputStream input(_input.get());

  // Read the size of the object.
  uint64_t size;
  if (!input.ReadVarint64(&size)) {
    throw std::invalid_argument("Error deserializing model.");
  }

  // Tell the stream not to read beyond that size.
  auto limit = input.PushLimit(size);

  // Parse the message.
  if (!object.MergeFromCodedStream(&input) || !input.ConsumedEntireMessage()) {
    throw std::invalid_argument("Error deserializing model.");
  }

  // Release the limit.
  input.PopLimit(limit);
}

}  // namespace thirdai::utils