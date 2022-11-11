#include "SamplingConfig.h"
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

namespace thirdai::bolt {

template <class Archive>
void SamplingConfig::serialize(Archive& archive) {
  (void)archive;
}

template <class Archive>
void DWTASamplingConfig::serialize(Archive& archive) {
  archive(cereal::base_class<SamplingConfig>(this), _num_tables,
          _hashes_per_table, _reservoir_size);
}

template <class Archive>
void FastSRPSamplingConfig::serialize(Archive& archive) {
  archive(cereal::base_class<SamplingConfig>(this), _num_tables,
          _hashes_per_table, _reservoir_size);
}

template <class Archive>
void RandomSamplingConfig::serialize(Archive& archive) {
  archive(cereal::base_class<SamplingConfig>(this));
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::DWTASamplingConfig)
CEREAL_REGISTER_TYPE(thirdai::bolt::FastSRPSamplingConfig)
CEREAL_REGISTER_TYPE(thirdai::bolt::RandomSamplingConfig)
