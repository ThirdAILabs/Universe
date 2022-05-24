#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <optional>
#include <utility>
#include <vector>

namespace thirdai::dataset {

using BoltDataLabelPair = std::pair<bolt::BoltBatch, bolt::BoltBatch>;

class BatchProcessor {
 public:
  std::optional<BoltDataLabelPair> createBatch(
      const std::vector<std::string>& rows) {
#pragma omp parallel for default(none) shared(rows)
    for (const auto& row : rows) {
      processRow(row);
    }

    // TODO(nicholas): does moving vector set it to empty?
    return std::make_pair(bolt::BoltBatch(std::move(_data_vecs)),
                          bolt::BoltBatch(std::move(_label_vecs)));
  }

  virtual ~BatchProcessor() = default;

 protected:
  void appendSample(bolt::BoltVector&& data_vec, bolt::BoltVector&& label_vec) {
    // TODO(nicholas): lock for concurrent access
    _data_vecs.push_back(std::move(data_vec));
    _label_vecs.push_back(std::move(label_vec));
  }

  virtual void processRow(const std::string& row) = 0;

 private:
  std::vector<bolt::BoltVector> _data_vecs;
  std::vector<bolt::BoltVector> _label_vecs;
};

}  // namespace thirdai::dataset