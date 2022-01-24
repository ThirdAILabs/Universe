#pragma once

#include <dataset/src/Factory.h>
#include <dataset/src/Vectors.h>
#include <cassert>
#include <exception>
#include <fstream>
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
  CsvDenseBatchFactory(char delimiter): _delimiter(delimiter) {
    if (delimiter == '.' 
        || delimiter == '0'
        || delimiter == '1'
        || delimiter == '2'
        || delimiter == '3'
        || delimiter == '4'
        || delimiter == '5'
        || delimiter == '6'
        || delimiter == '7'
        || delimiter == '8'
        || delimiter == '9') {
      throw std::invalid_argument("Invalid delimiter: " + delimiter);
    }
  }

  DenseBatch parse(std::ifstream& file, uint32_t target_batch_size,
                   uint64_t start_id) override {
    DenseBatch batch(start_id);

    uint32_t dim = 0;

    std::string line;
    while (batch._batch_size < target_batch_size && std::getline(file, line)) {
      
      if (line.size() == 0) {
        continue;
      }

      const char* start = line.c_str();
      const char* const line_end = line.c_str() + line.size();
      char* end;

      uint32_t label = std::strtoul(start, &end, 10);
      if (start == end) {
        throw std::invalid_argument("Invalid dataset file: Found a line that doesn't start with a label.");
      }
      batch._labels.push_back({label});
      if (line_end - end == 1) {
        throw std::invalid_argument("Invalid dataset file: The line only contains a label.");
      }
      start = end;
      std::vector<float> values;
      while (start < line_end) {
        if (*start != _delimiter) {
          throw std::invalid_argument("Invalid dataset file: Found invalid character: " + *start);
        }
        start++;
        if (start == line_end) {
          throw std::invalid_argument("Invalid dataset file: No number after delimiter.");
        }
        float value = std::strtof(start, &end);
        if (start == end) {
          throw std::invalid_argument("Invalid dataset file: Found invalid character: " + *start);
        }
        values.push_back(value);
        start = end;
      }

      if (dim != 0 && dim != values.size()) {
        throw std::invalid_argument(
            "Invalid dataset file: Contains different-dimensional vectors.\n");
      }

      dim = values.size();

      DenseVector v(values.size());
      uint32_t cnt = 0;
      for (const auto& x : values) {
        v._values[cnt] = x;
        cnt++;
      }

      batch._vectors.push_back(std::move(v));
      batch._batch_size++;
    }
    return batch;
  }

 private:
  char _delimiter;
};

}  // namespace thirdai::dataset
