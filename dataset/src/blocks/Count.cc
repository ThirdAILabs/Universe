#include "Count.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>

namespace thirdai::dataset {

template void CountBlock::serialize(cereal::BinaryInputArchive&);
template void CountBlock::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void CountBlock::serialize(Archive& archive) {
  archive(cereal::base_class<Block>(this), _column, _delimiter, _ceiling);
}
}  //  namespace thirdai::dataset
