#pragma once

#include <cereal/types/polymorphic.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/InputTypes.h>
#include <exceptions/src/Exceptions.h>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace thirdai::dataset {

class Featurizer {
 public:
  /**
   * Featurizes a list of input rows into N different "datasets". Each dataset
   * is simply a vector of BoltVectors. Each "dataset" should have the same
   * number of BoltVectors. BoltVectors with the same index are "corresponding",
   * and usually this means that they can be trained on as corresponding
   * examples.
   * TODO(Any): Consider transposing the vector of vector format here and
   * elsewhere for consistency
   */
  virtual std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& rows) = 0;

  virtual MapInputBatch convertToMapInputBatch(
      const LineInputBatch& input_batch, const std::string& output_column_name,
      const std::string& input_column_name, const std::string& header) {
    (void)input_batch;
    (void)output_column_name, (void)input_column_name;
    (void)header;
    throw exceptions::NotImplemented("Cannot convert to MapInputBatch");
  }

  virtual bool expectsHeader() const = 0;

  virtual void processHeader(const std::string& header) = 0;

  // Returns the size of the vector returned from featurizer, i.e. how many
  // datasets this featurizer featurizes rows into
  virtual size_t getNumDatasets() = 0;

  virtual ~Featurizer() = default;

  // Returns a vector of the BoltVector dimensions one would get if they called
  // featurize.
  virtual std::vector<uint32_t> getDimensions() {
    // By default we assume that this is an unsupported operation
    throw exceptions::NotImplemented(
        "Cannot get the dimensions for this featurizer");
  }
  // Default constructor for cereal.
  Featurizer() {}

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using FeaturizerPtr = std::shared_ptr<Featurizer>;

}  // namespace thirdai::dataset
