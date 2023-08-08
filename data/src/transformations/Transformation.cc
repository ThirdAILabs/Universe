#include "Transformation.h"
#include <data/src/transformations/protobuf_utils/FromProto.h>
#include <proto/transformations.pb.h>

namespace thirdai::data {

std::string Transformation::serialize() const {
  auto* proto = toProto();
  auto binary = proto->SerializeAsString();
  delete proto;

  return binary;
}

std::shared_ptr<Transformation> Transformation::deserialize(
    const std::string& binary) {
  proto::data::Transformation transformation;
  transformation.ParseFromString(binary);

  return fromProto(transformation);
}

}  // namespace thirdai::data