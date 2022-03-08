#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/parsers/SvmParser.h>

namespace thirdai::dataset {

using bolt::BoltVector;

class BoltInputBatch {
 public:
  BoltInputBatch(const std::vector<std::vector<uint32_t>>& indices,
                 const std::vector<std::vector<float>>& values,
                 const std::vector<std::vector<uint32_t>>& labels) {
    assert(indices.size() == values.size() && indices.size() == labels.size());

    for (uint32_t i = 0; i < indices.size(); i++) {
      _vectors.push_back(
          BoltVector::makeSparseVector(indices.at(i), values.at(i)));
      _labels.push_back(BoltVector::makeSparseVector(
          labels[i],
          std::vector<float>(labels[i].size(), 1.0 / labels[i].size())));
    }
  }

  BoltInputBatch(const std::vector<std::vector<float>>& values,
                 const std::vector<std::vector<uint32_t>>& labels) {
    assert(values.size() == labels.size());

    for (uint32_t i = 0; i < values.size(); i++) {
      _vectors.push_back(BoltVector::makeDenseVector(values.at(i)));
      _labels.push_back(BoltVector::makeSparseVector(
          labels[i],
          std::vector<float>(labels[i].size(), 1.0 / labels[i].size())));
    }
  }

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

 private:
  std::vector<BoltVector> _vectors;
  std::vector<BoltVector> _labels;
};

}  // namespace thirdai::dataset