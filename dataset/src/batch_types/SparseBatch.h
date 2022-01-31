#pragma once

#include <dataset/src/Factory.h>
#include <dataset/src/Vectors.h>
#include <cassert>
#include <fstream>
#include <vector>

namespace thirdai::dataset {

class SparseBatch {
  friend class SvmSparseBatchFactory;

 public:
  explicit SparseBatch(uint64_t start_id)
      : _batch_size(0), _start_id(start_id) {}
  // Take r-value reference for vectors to force a move.
  SparseBatch(std::vector<SparseVector>&& vectors,
              std::vector<std::vector<uint32_t>>&& labels, uint64_t start_id)
      : _vectors(std::move(vectors)),
        _batch_size(_vectors.size()),
        _labels(std::move(labels)),
        _start_id(start_id) {}

  /**
   * Explicitly constructs a sparse batch from a vector of SparseVectors and a
   * starting id. Note that this sets each items label vector to be empty.
   **/
  SparseBatch(std::vector<SparseVector>&& vectors, uint64_t start_id)
      : _vectors(std::move(vectors)),
        _batch_size(_vectors.size()),
        _labels(_vectors.size()),
        _start_id(start_id) {}

  const SparseVector& operator[](uint32_t i) const { return _vectors[i]; }

  const SparseVector& at(uint32_t i) const { return _vectors.at(i); }

  const std::vector<uint32_t>& labels(uint32_t i) const { return _labels[i]; }

  uint64_t id(uint32_t i) const { return _start_id + i; }

  uint32_t getBatchSize() const { return _batch_size; }

 private:
  std::vector<SparseVector> _vectors;
  uint32_t _batch_size;
  std::vector<std::vector<uint32_t>> _labels;
  uint64_t _start_id;
};

class SvmSparseBatchFactory : public Factory<SparseBatch> {
 public:
  SvmSparseBatchFactory() {}

  static void parseSVMBatch(std::ifstream& file, uint32_t target_batch_size,
                            std::vector<uint32_t>& indices,
                            std::vector<float>& values,
                            std::vector<uint32_t>& markers,
                            std::vector<std::vector<uint32_t>>& batch_labels) {
    uint32_t vectors_read = 0;
    std::string line;
    markers.push_back(0);
    while (vectors_read < target_batch_size && std::getline(file, line)) {
      const char* start = line.c_str();
      const char* const line_end = line.c_str() + line.size();
      char* end;
      std::vector<uint32_t> labels;
      do {
        uint32_t label = std::strtoul(start, &end, 10);
        labels.push_back(label);
        start = end;
      } while ((*start++) == ',');
      batch_labels.push_back(std::move(labels));

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

      markers.push_back(indices.size());
      vectors_read++;
    }
  }

  SparseBatch parse(std::ifstream& file, uint32_t target_batch_size,
                    uint64_t start_id) override {
    SparseBatch batch(start_id);

    std::vector<uint32_t> indices, markers;
    std::vector<float> values;
    std::vector<std::vector<uint32_t>> labels;

    parseSVMBatch(file, target_batch_size, indices, values, markers, labels);

    for (uint32_t i = 0; i < markers.size() - 1; i++) {
      uint32_t start = markers.at(i);
      uint32_t end = markers.at(i + 1);
      SparseVector vec(markers.at(i + 1) - start);
      std::copy(indices.begin() + start, indices.begin() + end, vec._indices);
      std::copy(values.begin() + start, values.begin() + end, vec._values);
      batch._vectors.push_back(std::move(vec));
    }
    batch._labels = std::move(labels);

    batch._batch_size = markers.size() - 1;

    return batch;
  }
};

}  // namespace thirdai::dataset
