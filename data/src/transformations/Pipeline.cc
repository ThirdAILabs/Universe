#include "Pipeline.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>

namespace thirdai::data {

void Pipeline::buildExplanationMap(const ColumnMap& input, State& state,
                                   ExplanationMap& explanations) const {
  ColumnMap last_input = input;

  for (const auto& transformation : _transformations) {
    // Apply the transformation first to make sure that the input is valid.
    ColumnMap next_input = transformation->apply(last_input, state);
    transformation->buildExplanationMap(last_input, state, explanations);
    last_input = std::move(next_input);
  }
}

ar::ConstArchivePtr Pipeline::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  auto transformations = ar::List::make();
  for (const auto& t : _transformations) {
    transformations->append(t->toArchive());
  }

  map->set("transformations", transformations);

  return map;
}

Pipeline::Pipeline(const ar::Archive& archive) {
  for (const auto& t : archive.get("transformations")->list()) {
    _transformations.push_back(Transformation::fromArchive(*t));
  }
}

void Pipeline::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void Pipeline::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

PipelinePtr Pipeline::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

PipelinePtr Pipeline::load_stream(std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  PipelinePtr deserialize_into(new Pipeline());
  iarchive(*deserialize_into);
  return deserialize_into;
}

template <class Archive>
void Pipeline::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _transformations);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::Pipeline)
