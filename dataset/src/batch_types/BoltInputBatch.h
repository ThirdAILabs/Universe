#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Factory.h>
#include <dataset/src/parsers/CsvParser.h>
#include <dataset/src/parsers/SvmParser.h>

namespace thirdai::dataset {

using bolt::BoltVector;

/**
 * This class is a dataset batch using the BoltVector data format. This differs
 * from the BoltBatch because it also includes the labels.
 */
class BoltInputBatch {
 public:
  BoltInputBatch(std::vector<BoltVector>&& vectors,
                 std::vector<BoltVector>&& labels)
      : _vectors(std::move(vectors)), _labels(std::move(labels)) {}
  
  BoltInputBatch() {}

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
    ss << "===================================================================="
          "====\n";
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
    ss << "===================================================================="
          "====";
    return ss.str();
  }

 private:
  std::vector<BoltVector> _vectors;
  std::vector<BoltVector> _labels;
};

class BoltSvmBatchFactory : public Factory<BoltInputBatch> {
 private:
  SvmParser<BoltVector, BoltVector> _parser;

 public:
  // We can use the SVM parser with takes in functions that construct the
  // desired vector/label format (in this case BoltVector) from vectors of
  // indices and values and the labels
  BoltSvmBatchFactory()
      : _parser(BoltVector::makeSparseVector,
                [](const std::vector<uint32_t>& labels) -> BoltVector {
                  return BoltVector::makeSparseVector(
                      labels,
                      std::vector<float>(labels.size(), 1.0 / labels.size()));
                }) {}

  BoltInputBatch parse(std::ifstream& file, uint32_t target_batch_size,
                       uint64_t start_id) override {
    (void)start_id;
    std::vector<BoltVector> vectors;
    std::vector<BoltVector> labels;

    _parser.parseBatch(target_batch_size, file, vectors, labels);

    return BoltInputBatch(std::move(vectors), std::move(labels));
  }
};

class BoltCsvBatchFactory : public Factory<BoltInputBatch> {
 private:
  CsvParser<BoltVector, BoltVector> _parser;

 public:
  // We can use the CSV parser with takes in functions that construct the
  // desired vector/label format (in this case BoltVector) from a vector of
  // values and the labels.
  explicit BoltCsvBatchFactory(char delimiter)
      : _parser(
            BoltVector::makeDenseVector,
            [](uint32_t label) -> BoltVector {
              return BoltVector::makeSparseVector({label}, {1.0});
            },
            delimiter) {}

  BoltInputBatch parse(std::ifstream& file, uint32_t target_batch_size,
                       uint64_t start_id) override {
    (void)start_id;
    std::vector<BoltVector> vectors;
    std::vector<BoltVector> labels;

    _parser.parseBatch(target_batch_size, file, vectors, labels);

    return BoltInputBatch(std::move(vectors), std::move(labels));
  }
};

}  // namespace thirdai::dataset