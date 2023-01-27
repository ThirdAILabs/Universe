#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Featurizer.h>

namespace thirdai::dataset {

class SvmFeaturizer final : public Featurizer {
 public:
  explicit SvmFeaturizer(bool softmax_for_multiclass = true)
      : _softmax_for_multiclass(softmax_for_multiclass) {}

  bool expectsHeader() const final { return false; }

  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& rows) final {
    std::vector<BoltVector> _data_vecs = std::vector<BoltVector>(rows.size());
    std::vector<BoltVector> _label_vecs = std::vector<BoltVector>(rows.size());

    // #pragma omp parallel for default(none) shared(rows)
    for (uint32_t row_id = 0; row_id < rows.size(); row_id++) {
      auto p = processRow(rows[row_id]);

      _data_vecs[row_id] = std::move(p.first);
      _label_vecs[row_id] = std::move(p.second);
    }

    return {std::move(_data_vecs), std::move(_label_vecs)};
  }

  void processHeader(const std::string& header) final { (void)header; }

  size_t getNumDatasets() final { return 2; }

  std::pair<BoltVector, BoltVector> processRow(const std::string& line) const {
    const char* start = line.c_str();
    const char* const line_end = line.c_str() + line.size();
    char* end;

    // Parse the labels. The labels are comma separated without spaces.
    // Ex: 3,4,13
    std::vector<uint32_t> labels;
    do {
      uint32_t label = std::strtoul(start, &end, 10);
      labels.push_back(label);
      start = end;
    } while ((*start++) == ',');

    float label_val = _softmax_for_multiclass ? 1.0 / labels.size() : 1.0;
    BoltVector labels_vec = BoltVector::makeSparseVector(
        labels, std::vector<float>(labels.size(), label_val));

    // Parse the vector itself. The elements are given in <index>:<value>
    // pairs with tabs or spaces between each pair. There should also be a
    // tab/space between the labels and first pair.
    std::vector<uint32_t> indices;
    std::vector<float> values;
    do {
      uint32_t index = std::strtoul(start, &end, 10);
      start = end + 1;
      float value = std::strtof(start, &end);
      indices.push_back(index);
      values.push_back(value);
      start = end;

      while ((*start == ' ' || *start == '\t') && start < line_end) {
        start++;
      }
    } while (*start != '\n' && start < line_end);

    BoltVector data_vec = BoltVector::makeSparseVector(indices, values);

    return std::make_pair(std::move(data_vec), std::move(labels_vec));
  }

 private:
  bool _softmax_for_multiclass;
};

}  // namespace thirdai::dataset
