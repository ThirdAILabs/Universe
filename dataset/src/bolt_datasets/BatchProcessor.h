#pragma once

#include <cereal/types/polymorphic.hpp>
#include <bolt/src/layers/BoltVector.h>
#include <optional>
#include <utility>
#include <vector>

#ifndef __clang__
#include <omp.h>
#endif

namespace thirdai::dataset {

using BoltDataLabelPair = std::pair<bolt::BoltBatch, bolt::BoltBatch>;

class BatchProcessor {
 public:
  virtual std::optional<BoltDataLabelPair> createBatch(
      const std::vector<std::string>& rows) = 0;

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

class UnaryBatchProcessor : public BatchProcessor {
 public:
  std::optional<BoltDataLabelPair> createBatch(
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

    // TODO(nicholas): does moving vector set it to empty?
    return std::make_pair(bolt::BoltBatch(std::move(_data_vecs)),
                          bolt::BoltBatch(std::move(_label_vecs)));
  }

 protected:
  virtual std::pair<bolt::BoltVector, bolt::BoltVector> processRow(
      const std::string& row) = 0;

  // Default constructor for cereal.
  UnaryBatchProcessor() {}

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<BatchProcessor>(this));
  }
};

class NarayBatchProcesor : public BatchProcessor {
 public:
  NarayBatchProcesor() {
#ifndef __clang__
    omp_init_lock(&this->_lock);
#endif
  }

  std::optional<BoltDataLabelPair> createBatch(
      const std::vector<std::string>& rows) final {
    _data_vecs = {};
    _label_vecs = {};
#pragma omp parallel for default(none) shared(rows)
    for (const auto& row : rows) {
      processRow(row);
    }

    // TODO(nicholas): does moving vector set it to empty?
    return std::make_pair(bolt::BoltBatch(std::move(_data_vecs)),
                          bolt::BoltBatch(std::move(_label_vecs)));
  }

  virtual ~NarayBatchProcesor() {
#ifndef __clang__
    omp_destroy_lock(&this->_lock);
#endif
  }

 protected:
  void appendSample(bolt::BoltVector&& data_vec, bolt::BoltVector&& label_vec) {
#ifndef __clang__
    omp_set_lock(&this->_lock);
#endif
    _data_vecs.push_back(std::move(data_vec));
    _label_vecs.push_back(std::move(label_vec));
#ifndef __clang__
    omp_unset_lock(&this->_lock);
#endif
  }

  // Process row can call append sample when it generates samples based off of
  // the input row.
  virtual void processRow(const std::string& row) = 0;

 private:
  std::vector<bolt::BoltVector> _data_vecs;
  std::vector<bolt::BoltVector> _label_vecs;
#ifndef __clang__
  omp_lock_t _lock;
#endif
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::BatchProcessor);
CEREAL_REGISTER_TYPE(thirdai::dataset::UnaryBatchProcessor);
