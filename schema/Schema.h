#pragma once

#include <dataset/src/Dataset.h>
#include <dataset/src/Vectors.h>
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/batch_types/BoltInputBatch.h>
#include <dataset/src/batch_types/SparseBatch.h>
#include <sys/types.h>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <iomanip>
#include "InProgressVector.h"
#include "Date.h"

namespace thirdai::schema {

struct ABlock {
  virtual void extractFeatures(std::vector<std::string_view> line, InProgressSparseVector& output_vec) = 0;
};

struct ABlockConfig {
  virtual size_t maxColumn() const = 0;
  virtual size_t featureDim() const = 0;
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
  DataLoader(std::vector<std::shared_ptr<ABlockConfig>>& input_block_configs, std::vector<std::shared_ptr<ABlockConfig>>& label_block_configs, uint32_t batch_size)
  : _batch_size(batch_size) {
    buildBlocks(input_block_configs, _input_blocks, _input_dim);
    buildBlocks(label_block_configs, _label_blocks, _label_dim);
  }

  size_t inputDim() const { return _input_dim; }
  
  size_t labelDim() const { return _label_dim; }

  void readCSV(const std::string& filename, const char delimiter) {
    if (_in.is_open()) {
      _in.close();
    }
    _in.open(filename);
    _delimiter = delimiter;
    if (!_in.is_open()) {
      std::stringstream ss;
      ss << "File not found: " << filename << std::endl;
      throw std::invalid_argument(ss.str());
    }
  }

  void processCSV(uint32_t max_lines=std::numeric_limits<uint32_t>::max()) {
    std::string line;
    uint32_t i = 0;
    while (i < max_lines && std::getline(_in, line)) {
      consumeCSVLine(line, _delimiter);
      i++;
    }
  }

  // I want to consider how a line, say, from a CSV, is consumed.
  void consumeCSVLine(std::string_view line, char delimiter) {
    size_t start = 0;
    for (size_t i = 0; i < _line_length; ++i) {
      auto end = line.find(delimiter, start);
      _line_buf[i] = line.substr(start, end - start);
      start = end + 1;
    }
    
    addVector(_line_buf, _input_blocks, _input_vectors);
    addVector(_line_buf, _label_blocks, _label_vectors);
  }

  dataset::InMemoryDataset<dataset::BoltInputBatch> exportDataset(size_t max_exported=std::numeric_limits<size_t>::max(), bool shuffle=true) {
    if (!_in.is_open()) {
      std::stringstream ss;
      throw std::invalid_argument("Error: readCSV() not called before exportDataset()");
    }

    processCSV(max_exported);

    assert(_input_vectors.size() == _label_vectors.size());
    uint32_t n_exported = _input_vectors.size();

    std::vector<uint32_t> positions(n_exported);
    for (uint32_t i = 0; i < n_exported; i++) {
      positions[i] = i;
    }

    if (shuffle) {
      auto rng = std::default_random_engine {};
      std::shuffle(positions.begin(), positions.end(), rng);
    }

    std::vector<dataset::BoltInputBatch> batches;

    for (uint32_t batch_start_index = 0; batch_start_index < n_exported; batch_start_index += _batch_size) {
      std::vector<bolt::BoltVector> batch_inputs;
      std::vector<bolt::BoltVector> batch_labels;
      for (uint32_t index_in_batch = 0; index_in_batch < std::min(_batch_size, n_exported - batch_start_index); index_in_batch++) {
        batch_inputs.push_back(std::move(_input_vectors[positions[batch_start_index + index_in_batch]]));
        batch_labels.push_back(std::move(_label_vectors[positions[batch_start_index + index_in_batch]]));
      }
      batches.emplace_back(std::move(batch_inputs), std::move(batch_labels));
    }

    _input_vectors = std::vector<bolt::BoltVector>();
    _label_vectors = std::vector<bolt::BoltVector>();

    std::cout << "Exporting Bolt dataset with " << n_exported << " elements." << std::endl;

    // Then move to dataset
    return { std::move(batches), n_exported };
  }

 private:

  void buildBlocks(std::vector<std::shared_ptr<ABlockConfig>>& block_configs, std::vector<std::unique_ptr<ABlock>>& blocks, size_t& feature_dim) {
    uint32_t offset = 0;
    for (const auto& config : block_configs) {
      blocks.push_back(config->build(offset));
      _line_length = std::max(_line_length, config->maxColumn() + 1); // Columns are 0-indexed.
      feature_dim += config->featureDim();
    }
    _line_buf.resize(_line_length);
  }

  static void addVector(std::vector<std::string_view>& line_buf, std::vector<std::unique_ptr<ABlock>>& blocks, std::vector<bolt::BoltVector>& vectors) {
    InProgressSparseVector vec;
    for (auto& block : blocks) {
      block->extractFeatures(line_buf, vec);
    }
    vectors.push_back(vec.toBoltVector());
  }

  
  char _delimiter = ',';
  size_t _input_dim = 0;
  size_t _label_dim = 0;
  size_t _line_length = 0;
  uint32_t _batch_size;
  std::ifstream _in;
  std::vector<bolt::BoltVector> _input_vectors;
  std::vector<bolt::BoltVector> _label_vectors;
  std::vector<std::string_view> _line_buf; // TODO(Geordie): Initialize
  std::vector<std::unique_ptr<ABlock>> _input_blocks;
  std::vector<std::unique_ptr<ABlock>> _label_blocks;
};

} // namespace thirdai::schema