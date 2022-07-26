#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/BatchProcessor.h>
#include <dataset/src/batch_types/MaskedSentenceBatch.h>

namespace thirdai::dataset {

class SvmBatchProcessor final : public UnaryBoltBatchProcessor {
 public:
  explicit SvmBatchProcessor(bool softmax_for_multiclass = true)
      : _softmax_for_multiclass(softmax_for_multiclass) {}

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

  std::pair<bolt::BoltVector, bolt::BoltVector> processRow(
      const std::string& line) final {
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