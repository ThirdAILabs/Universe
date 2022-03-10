#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Factory.h>
#include <dataset/src/parsers/CsvParser.h>
#include <dataset/src/parsers/SvmParser.h>

namespace thirdai::dataset {

using bolt::BoltVector;

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

 private:
  std::vector<BoltVector> _vectors;
  std::vector<BoltVector> _labels;
};

class BoltSvmBatchFactory : public Factory<BoltInputBatch> {
 private:
  SvmParser<BoltVector, BoltVector> _parser;

 public:
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