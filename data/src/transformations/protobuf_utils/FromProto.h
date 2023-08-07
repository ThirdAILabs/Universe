#pragma once

#include <data/src/transformations/Transformation.h>
#include <proto/transformations.pb.h>

namespace thirdai::data {

TransformationPtr fromProto(const proto::data::Transformation& transformation);

}  // namespace thirdai::data
