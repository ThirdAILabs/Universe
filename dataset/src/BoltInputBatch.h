#pragma once

#include "BoltVector.h"
#include <string>
#include <sstream>

namespace thirdai::dataset {

// TODO(Geordie): Bolt input batch must also support lack of labels.

class BoltInputBatch {
 public:
  BoltInputBatch(std::vector<BoltVector>&& vectors,
                 std::vector<BoltVector>&& labels)
      : _vectors(std::move(vectors)), _labels(std::move(labels)) {}

  uint32_t getBatchSize() const { return _vectors.size(); }

  const BoltVector& operator[](size_t i) const {
    assert(i < _vectors.size());
    return _vectors[i];
  }

  BoltVector& operator[](size_t i) {
    assert(i < _vectors.size());
    return _vectors[i];
  }

  const BoltVector& labels(size_t i) const {
    assert(i < _labels.size());
    return _labels[i];
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "========================================================================\n";
    ss << "Batch | size = " << _vectors.size() << "\n\n";
    if (_labels.size() == _vectors.size()) {
      for (size_t i = 0; i < _vectors.size(); i++) {
        ss << "Vector " << i << ":\n";
        ss << "Input: " << _vectors.at(i).toString() << "\n";
        ss << "Target: " << _labels.at(i).toString() << "\n";
      }
    } else {
      for (size_t i = 0; i < _vectors.size(); i++) {
        ss << "Vector " << i << ": " << _vectors.at(i).toString() << "\n\n";
      }
    }
    ss << "========================================================================";
    return ss.str();
  }

 private:
  std::vector<BoltVector> _vectors;
  std::vector<BoltVector> _labels;
};

}  // namespace thirdai::dataset