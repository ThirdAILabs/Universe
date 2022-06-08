#pragma once

#include <cereal/types/polymorphic.hpp>
#include <bolt/src/layers/BoltVector.h>
#include <optional>
#include <utility>
#include <vector>

namespace thirdai::dataset {

template <typename BATCH_T>
using BoltDataLabelPair = std::pair<BATCH_T, bolt::BoltBatch>;

template <typename BATCH_T>
class BatchProcessor {
 public:
  virtual std::optional<BoltDataLabelPair<BATCH_T>> createBatch(
      const std::vector<std::string>& rows) = 0;

  virtual bool expectsHeader() const = 0;

  virtual void processHeader(const std::string& header) = 0;

  virtual ~BatchProcessor() = default;

 protected:
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

class UnaryBoltBatchProcessor : public BatchProcessor<bolt::BoltBatch> {
 public:
  std::optional<BoltDataLabelPair<bolt::BoltBatch>> createBatch(
      const std::vector<std::string>& rows) final {
    std::vector<bolt::BoltVector> _data_vecs =
        std::vector<bolt::BoltVector>(rows.size());
    std::vector<bolt::BoltVector> _label_vecs =
        std::vector<bolt::BoltVector>(rows.size());

    // #pragma omp parallel for default(none) shared(rows)
    for (uint32_t row_id = 0; row_id < rows.size(); row_id++) {
      auto p = processRow(rows[row_id]);

      _data_vecs[row_id] = std::move(p.first);
      _label_vecs[row_id] = std::move(p.second);
    }

    return std::make_pair(bolt::BoltBatch(std::move(_data_vecs)),
                          bolt::BoltBatch(std::move(_label_vecs)));
  }

 protected:
  virtual std::pair<bolt::BoltVector, bolt::BoltVector> processRow(
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

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::BatchProcessor<thirdai::bolt::BoltBatch>)
CEREAL_REGISTER_TYPE(thirdai::dataset::UnaryBoltBatchProcessor)
