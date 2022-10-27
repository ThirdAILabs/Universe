#pragma once

#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <algorithm>
#include <memory>

namespace thirdai::dataset {

class FeaturizationPipeline;
using FeaturizationPipelinePtr = std::shared_ptr<FeaturizationPipeline>;

class FeaturizationPipeline {
 public:
  explicit FeaturizationPipeline(std::vector<TransformationPtr> transformations)
      : _transformations(std::move(transformations)) {}

  ColumnMap featurize(ColumnMap columns) {
    for (auto& transformation : _transformations) {
      transformation->apply(columns);
    }

    return columns;
  }

  void save(const std::string& filename) const {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    save_stream(filestream);
  }

  void save_stream(std::ostream& output_stream) const {
    cereal::BinaryOutputArchive oarchive(output_stream);
    oarchive(*this);
  }

  static FeaturizationPipelinePtr load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    return load_stream(filestream);
  }

  static FeaturizationPipelinePtr load_stream(std::istream& input_stream) {
    cereal::BinaryInputArchive iarchive(input_stream);
    std::shared_ptr<FeaturizationPipeline> deserialize_into(
        new FeaturizationPipeline());
    iarchive(*deserialize_into);
    return deserialize_into;
  }

 private:
  std::vector<TransformationPtr> _transformations;

  // Private constructor for cereal.
  FeaturizationPipeline(){};

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_transformations);
  }
};

}  // namespace thirdai::dataset