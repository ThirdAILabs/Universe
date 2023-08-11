#include "TransformationList.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <data/src/ColumnMap.h>

namespace thirdai::data {

void TransformationList::explainFeatures(
    const ColumnMap& input, State& state,
    FeatureExplainations& explainations) const {
  ColumnMap last_input = input;

  for (const auto& transformation : _transformations) {
    // Apply the transformatino first to make sure that the input is valid.
    ColumnMap next_input = transformation->apply(last_input, state);
    transformation->explainFeatures(last_input, state, explainations);
    last_input = std::move(next_input);
  }
}

void TransformationList::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void TransformationList::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

TransformationListPtr TransformationList::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

TransformationListPtr TransformationList::load_stream(
    std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<TransformationList> deserialize_into(
      new TransformationList());
  iarchive(*deserialize_into);
  return deserialize_into;
}

template <class Archive>
void TransformationList::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _transformations);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::TransformationList)
