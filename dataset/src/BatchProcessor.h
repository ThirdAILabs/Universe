#pragma once

#include <cereal/types/polymorphic.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <exceptions/src/Exceptions.h>
#include <optional>
#include <utility>
#include <vector>

namespace thirdai::dataset {

class BatchProcessor {
 public:
  virtual std::vector<BoltBatch> createBatch(
      const std::vector<std::string>& rows) = 0;

  virtual bool expectsHeader() const = 0;

  virtual void processHeader(const std::string& header) = 0;

  virtual ~BatchProcessor() = default;

  // Returns a vector of the BoltVector dimensions one would get if they called
  // createBatch.
  virtual std::vector<uint32_t> getDimensions() {
    // By default we assume that this is an unsupported operation
    throw exceptions::NotImplemented(
        "Cannot get the dimensions for this batch processor");
  }
  // Default constructor for cereal.
  BatchProcessor() {}

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using BatchProcessorPtr = std::shared_ptr<BatchProcessor>;

}  // namespace thirdai::dataset
