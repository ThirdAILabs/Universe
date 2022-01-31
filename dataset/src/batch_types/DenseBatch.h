#pragma once

#include <dataset/src/Factory.h>
#include <dataset/src/Vectors.h>
#include <cassert>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace thirdai::dataset {

class DenseBatch {
  friend class CsvDenseBatchFactory;

 public:
  explicit DenseBatch(uint64_t start_id)
      : _batch_size(0), _start_id(start_id) {}

  // Take r-value reference for vectors to force a move.
  DenseBatch(std::vector<DenseVector>&& vectors,
             std::vector<std::vector<uint32_t>>&& labels, uint64_t start_id)
      : _vectors(std::move(vectors)),
        _batch_size(_vectors.size()),
        _labels(std::move(labels)),
        _start_id(start_id) {}

  /**
   * Explicitly constructs a dense batch from a vector of DenseVectors and a
   * starting id. Note that this sets each items label vector to be empty.
   **/
  DenseBatch(std::vector<DenseVector>&& vectors, uint64_t start_id)
      : _vectors(std::move(vectors)),
        _batch_size(_vectors.size()),
        _labels(_vectors.size()),
        _start_id(start_id) {}

  const DenseVector& operator[](uint32_t i) const { return _vectors[i]; };

  const DenseVector& at(uint32_t i) const { return _vectors.at(i); }

  const std::vector<uint32_t>& labels(uint32_t i) const { return _labels[i]; }

  uint64_t id(uint32_t i) const { return _start_id + i; }

  uint32_t getBatchSize() const { return _batch_size; }

 private:
  std::vector<DenseVector> _vectors;
  uint32_t _batch_size;
  std::vector<std::vector<uint32_t>> _labels;
  uint64_t _start_id;
};

class CsvDenseBatchFactory : public Factory<DenseBatch> {
 public:
  explicit CsvDenseBatchFactory(char delimiter) : _delimiter(delimiter) {
    if (delimiter == '.' || (delimiter >= '0' && delimiter <= '9')) {
      throw std::invalid_argument("Invalid delimiter: " + delimiter);
    }
  }

  static uint32_t parseCSVBatch(std::ifstream& file, uint32_t target_batch_size,
                                std::vector<float>& values,
                                std::vector<std::vector<uint32_t>>& labels,
                                char delimiter) {
    uint32_t dim = 0;
    uint32_t curr_batch_size = 0;
    std::string line;
    while (curr_batch_size < target_batch_size && std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }

      const char* start = line.c_str();
      const char* const line_end = line.c_str() + line.size();
      char* end;

      uint32_t label = std::strtoul(start, &end, 10);
      if (start == end) {
        throw std::invalid_argument(
            "Invalid dataset file: Found a line that doesn't start with a "
            "label.");
      }
      labels.push_back({label});
      if (line_end - end < 1) {
        throw std::invalid_argument(
            "Invalid dataset file: The line only contains a label.");
      }
      start = end;
      uint32_t curr_dim = 0;
      while (start < line_end) {
        if (*start != delimiter) {
          std::stringstream error_ss;
          error_ss << "Invalid dataset file: Found invalid character: "
                   << *start;
          throw std::invalid_argument(error_ss.str());
        }
        start++;
        if (start == line_end) {
          values.push_back(0);
        } else {
          float value = std::strtof(start, &end);
          if (start == end && *start != delimiter) {
            std::stringstream error_ss;
            error_ss << "Invalid dataset file: Found invalid character: "
                     << *start;
            throw std::invalid_argument(error_ss.str());
          }
          // value defaults to 0, So if start == end but start == delimiter,
          // value = 0.
          values.push_back(value);
          start = end;
        }
        curr_dim++;
      }

      if (dim != 0 && dim != curr_dim) {
        throw std::invalid_argument(
            "Invalid dataset file: Contains different-dimensional vectors.\n");
      }

      dim = curr_dim;

      DenseVector v(values.size());
      uint32_t cnt = 0;
      for (const auto& x : values) {
        v._values[cnt] = x;
        cnt++;
      }

      curr_batch_size++;
    }
    return curr_batch_size;
  }

  DenseBatch parse(std::ifstream& file, uint32_t target_batch_size,
                   uint64_t start_id) override {
    DenseBatch batch(start_id);

    std::vector<float> values;
    std::vector<std::vector<uint32_t>> labels;
    batch._batch_size =
        parseCSVBatch(file, target_batch_size, values, labels, _delimiter);

    uint32_t dim = values.size() / batch._batch_size;

    for (uint32_t i = 0; i < batch._batch_size; i++) {
      DenseVector vec(dim);
      std::copy(values.data() + i * dim, values.data() + (i + 1) * dim,
                vec._values);
      batch._vectors.push_back(std::move(vec));
    }
    batch._labels = std::move(labels);

    return batch;
  }

 private:
  char _delimiter;
};

}  // namespace thirdai::dataset
