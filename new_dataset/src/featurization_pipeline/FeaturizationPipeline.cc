#include "FeaturizationPipeline.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

namespace thirdai::data {

void FeaturizationPipeline::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void FeaturizationPipeline::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

FeaturizationPipelinePtr FeaturizationPipeline::load(
    const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

FeaturizationPipelinePtr FeaturizationPipeline::load_stream(
    std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<FeaturizationPipeline> deserialize_into(
      new FeaturizationPipeline());
  iarchive(*deserialize_into);
  return deserialize_into;
}

template <class Archive>
void FeaturizationPipeline::serialize(Archive& archive) {
  archive(_transformations);
}

}  // namespace thirdai::data
