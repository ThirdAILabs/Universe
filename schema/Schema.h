#pragma once

#include <dataset/src/Dataset.h>
#include <dataset/src/Vectors.h>
#include <dataset/src/batch_types/SparseBatch.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <iomanip>
#include "InProgressVector.h"
#include "Date.h"

namespace thirdai::schema {

struct ABlock {
  virtual void consume(std::vector<std::string_view> line, InProgressVector& output_vec) = 0;
};

struct ABlockBuilder {
  virtual size_t maxColumn() const = 0;
  virtual size_t inputFeatDim() const = 0;
  virtual std::unique_ptr<ABlock> build(uint32_t& offset) const = 0;
};

inline static uint32_t getNumberU32(std::string_view str) {
  const char* start = str.cbegin();
  char* end;
  return std::strtoul(start, &end, 10);
}

inline static std::tm getTm(std::string_view str, const std::string& timestamp_fmt) {
  std::tm t = {};
  std::istringstream ss(std::string(str), std::ios_base::in);
  ss >> std::get_time(&t, timestamp_fmt.c_str());
  return t;
}

inline static uint32_t getSecondsSinceEpochU32(std::string_view str, const std::string& timestamp_fmt) {
  std::tm t = {};
  std::istringstream ss(std::string(str), std::ios_base::in);
  ss >> std::get_time(&t, timestamp_fmt.c_str());
  return mktime(&t);
}

const uint32_t SECONDS_IN_DAY = 60 * 60 * 24;


class DataLoader {
 public:
  DataLoader(std::vector<std::shared_ptr<ABlockBuilder>>& schema, uint32_t batch_size)
  : _batch_size(batch_size) {
    
    _vectors_for_next_batch.reserve(batch_size);
    _labels_for_next_batch.reserve(batch_size);

    uint32_t offset = 0;
    for (const auto& builder : schema) {
      _blocks.push_back(builder->build(offset));
      _line_length = std::max(_line_length, builder->maxColumn() + 1); // Columns are 0-indexed.
      _input_feat_dim += builder->inputFeatDim();
    }

    _line_buf.resize(_line_length);
  }

  size_t inputFeatDim() const { return _input_feat_dim; }

  void readCSV(const std::string& filename, const char delimiter) {
    std::ifstream in(filename);
    std::string line;
    if (!in.is_open()) {
      std::stringstream ss;
      ss << "File not found: " << filename << std::endl;
      throw std::invalid_argument(ss.str());
    }
    while (std::getline(in, line)) {
      consumeCSVLine(line, delimiter);
    }
  }

  // I want to consider how a line, say, from a CSV, is consumed.
  void consumeCSVLine(std::string_view line, char delimiter) {
    _in_progress_vec.clear(); // Clear previous vector, if any.

    size_t start = 0;
    for (size_t i = 0; i < _line_length; ++i) {
      auto end = line.find(delimiter, start);
      _line_buf[i] = line.substr(start, end - start);
      start = end + 1;
    }
    
    
    for (auto& block : _blocks) {
      block->consume(_line_buf, _in_progress_vec);
    }
    addToDataset();
  }

  dataset::InMemoryDataset<dataset::SparseBatch> exportDataset() {
    auto n_elems_in_batches = _batches.size() * _batch_size;
    auto n_elems_in_remaining_batch = _vectors_for_next_batch.size();
    // Move tail of dataset to batch
    if (!_vectors_for_next_batch.empty()) {
      _batches.emplace_back(std::move(_vectors_for_next_batch), std::move(_labels_for_next_batch), n_elems_in_batches);
      _vectors_for_next_batch = std::vector<dataset::SparseVector>();
      _vectors_for_next_batch.reserve(_batch_size);
      _labels_for_next_batch = std::vector<std::vector<uint32_t>>();
      _labels_for_next_batch.reserve(_batch_size);
    }

    std::cout << "Exporting " << n_elems_in_batches + n_elems_in_remaining_batch << " vectors in the dataset." << std::endl;

    // Then move to dataset
    auto dataset = dataset::InMemoryDataset<dataset::SparseBatch>(std::move(_batches), n_elems_in_batches + n_elems_in_remaining_batch);
    _batches = std::vector<dataset::SparseBatch>();
    return dataset;
  }

 private:
  void addToDataset() {
    dataset::SparseVector vec(_in_progress_vec.size(), _in_progress_vec.begin(), _in_progress_vec.end());
    _vectors_for_next_batch.push_back(std::move(vec));
    _labels_for_next_batch.emplace_back(_in_progress_vec.labels()); // labels() returns an L-value so this invokes copy constructor.
    if (!(_vectors_for_next_batch.size() % _batch_size)) {
      auto n_elems = _batches.size() * _batch_size;
      _batches.emplace_back(std::move(_vectors_for_next_batch), std::move(_labels_for_next_batch), n_elems);
      _vectors_for_next_batch = std::vector<dataset::SparseVector>();
      _vectors_for_next_batch.reserve(_batch_size);
      _labels_for_next_batch = std::vector<std::vector<uint32_t>>();
      _labels_for_next_batch.reserve(_batch_size);
    }
  }
  
  size_t _input_feat_dim = 0;
  size_t _line_length = 0;
  uint32_t _batch_size;
  InProgressVector _in_progress_vec;
  std::vector<dataset::SparseVector> _vectors_for_next_batch;
  std::vector<std::vector<uint32_t>> _labels_for_next_batch;
  std::vector<dataset::SparseBatch> _batches;
  std::vector<std::string_view> _line_buf; // TODO(Geordie): Initialize
  std::vector<std::unique_ptr<ABlock>> _blocks;
};

} // namespace thirdai::schema