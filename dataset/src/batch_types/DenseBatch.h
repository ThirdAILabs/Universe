#pragma once

#include <dataset/src/Factory.h>
#include <dataset/src/Vectors.h>
#include <dataset/src/parsers/CsvParser.h>
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
 private:
  CsvParser<DenseVector, std::vector<uint32_t>> _parser;

 public:
  // We can use the CSV parser with takes in functions that construct the
  // desired vector/label format (in this case DenseVector and a regular
  // vector) from a vector of alues and the labels.
  explicit CsvDenseBatchFactory(char delimiter)
      : _parser(
            [](const std::vector<float>& values) -> DenseVector {
              DenseVector vec(values.size());
              std::copy(values.begin(), values.end(), vec._values);
              return vec;
            },
            [](uint32_t label) -> std::vector<uint32_t> { return {label}; },
            delimiter) {
    if (delimiter == '.' || (delimiter >= '0' && delimiter <= '9')) {
      std::string msg = "Invalid delimiter: ";
      msg.push_back(delimiter);
      throw std::invalid_argument(msg.c_str());
    }
  }

  DenseBatch parse(std::ifstream& file, uint32_t target_batch_size,
                   uint64_t start_id) override {
    std::vector<DenseVector> vectors;
    std::vector<std::vector<uint32_t>> labels;

    _parser.parseBatch(target_batch_size, file, vectors, labels);

    return DenseBatch(std::move(vectors), std::move(labels), start_id);
  }
};

}  // namespace thirdai::dataset
