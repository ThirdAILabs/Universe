#pragma once

#include <cereal/types/polymorphic.hpp>
#include <bolt_vector/src/BoltVector.h>
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

class UnaryBoltBatchProcessor : public BatchProcessor {
 public:
  std::vector<BoltBatch> createBatch(
      const std::vector<std::string>& rows) final {
    std::vector<BoltVector> _data_vecs = std::vector<BoltVector>(rows.size());
    std::vector<BoltVector> _label_vecs = std::vector<BoltVector>(rows.size());

    // #pragma omp parallel for default(none) shared(rows)
    for (uint32_t row_id = 0; row_id < rows.size(); row_id++) {
      auto p = processRow(rows[row_id]);

      _data_vecs[row_id] = std::move(p.first);
      _label_vecs[row_id] = std::move(p.second);
    }

    return {BoltBatch(std::move(_data_vecs)),
            BoltBatch(std::move(_label_vecs))};
  }

 protected:
  virtual std::pair<BoltVector, BoltVector> processRow(
      const std::string& row) = 0;

  // Default constructor for cereal.
  UnaryBoltBatchProcessor() {}

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<BatchProcessor>(this));
  }
};

/**
 * This BatchProcessor provides an interface to compute metadata about a dataset
 * in a streaming fashion without creating BoltVectors
 */
class ComputeBatchProcessor : public BatchProcessor {
 public:
  std::vector<BoltBatch> createBatch(
      const std::vector<std::string>& rows) final {
    // TODO(david) enable parallel by making metadata calculation thread safe
    // #pragma omp parallel for default(none) shared(rows)
    for (const std::string& row : rows) {
      processRow(row);
    }

    return {BoltBatch(), BoltBatch()};
  }

 protected:
  virtual void processRow(const std::string& row) = 0;

  // Default constructor for cereal.
  ComputeBatchProcessor() {}

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<BatchProcessor>(this));
  }
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::UnaryBoltBatchProcessor)
CEREAL_REGISTER_TYPE(thirdai::dataset::ComputeBatchProcessor)
