#include "BoltDatasets.h"
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <dataset/src/parsers/ClickThroughParser.h>
#include <dataset/src/parsers/CsvParser.h>
#include <dataset/src/parsers/SvmParser.h>
#include <chrono>
#include <fstream>

namespace thirdai::dataset {

DatasetWithLabels loadBoltSvmDataset(const std::string& filename,
                                     uint32_t batch_size,
                                     bool labels_sum_to_one) {
  std::cout << "Loading Bolt SVM dataset from '" << filename << "' ..."
            << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  std::ifstream file(filename);

  SvmParser<bolt::BoltVector, bolt::BoltVector> parser(
      bolt::BoltVector::makeSparseVector,
      [labels_sum_to_one](
          const std::vector<uint32_t>& labels) -> bolt::BoltVector {
        float label_vals = labels_sum_to_one ? 1.0 / labels.size() : 1.0;
        return bolt::BoltVector::makeSparseVector(
            labels, std::vector<float>(labels.size(), label_vals));
      });

  uint32_t len = 0;

  std::vector<bolt::BoltBatch> data_batches;
  std::vector<bolt::BoltBatch> label_batches;
  while (!file.eof()) {
    std::vector<bolt::BoltVector> data_vecs;
    std::vector<bolt::BoltVector> label_vecs;

    parser.parseBatch(batch_size, file, data_vecs, label_vecs);

    len += data_vecs.size();

    data_batches.push_back(bolt::BoltBatch(std::move(data_vecs)));
    label_batches.push_back(bolt::BoltBatch(std::move(label_vecs)));
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::cout
      << " -> Read " << len << " vectors from '" << filename << "' in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;

  return DatasetWithLabels(BoltDataset(std::move(data_batches), len),
                           BoltDataset(std::move(label_batches), len));
}

DatasetWithLabels loadBoltCsvDataset(const std::string& filename,
                                     uint32_t batch_size, char delimiter) {
  std::cout << "Loading Bolt CSV dataset from '" << filename << "' ..."
            << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  std::ifstream file(filename);

  CsvParser<bolt::BoltVector, bolt::BoltVector> parser(
      bolt::BoltVector::makeDenseVector,
      [](uint32_t label) -> bolt::BoltVector {
        return bolt::BoltVector::makeSparseVector({label}, {1.0});
      },
      delimiter);

  uint32_t len = 0;

  std::vector<bolt::BoltBatch> data_batches;
  std::vector<bolt::BoltBatch> label_batches;
  while (!file.eof()) {
    std::vector<bolt::BoltVector> data_vecs;
    std::vector<bolt::BoltVector> label_vecs;

    parser.parseBatch(batch_size, file, data_vecs, label_vecs);

    len += data_vecs.size();

    data_batches.push_back(bolt::BoltBatch(std::move(data_vecs)));
    label_batches.push_back(bolt::BoltBatch(std::move(label_vecs)));
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::cout
      << " -> Read " << len << " vectors from '" << filename << "' in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;

  return DatasetWithLabels(BoltDataset(std::move(data_batches), len),
                           BoltDataset(std::move(label_batches), len));
}

ClickThroughDatasetWithLabels loadClickThroughDataset(
    const std::string& filename, uint32_t batch_size,
    uint32_t num_dense_features, uint32_t num_categorical_features,
    bool sparse_labels) {
  std::cout << "Loading click through dataset from '" << filename << "' ..."
            << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  std::ifstream file(filename);

  ClickThroughParser parser(num_dense_features, num_categorical_features,
                            sparse_labels);

  uint32_t len = 0;

  std::vector<ClickThroughBatch> data_batches;
  std::vector<bolt::BoltBatch> label_batches;
  while (!file.eof()) {
    auto [data, labels] = parser.parseBatch(batch_size, file);
    len += data.getBatchSize();

    data_batches.push_back(std::move(data));
    label_batches.push_back(bolt::BoltBatch(std::move(labels)));
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::cout
      << " -> Read " << len << " vectors from '" << filename << "' in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;

  return ClickThroughDatasetWithLabels(
      InMemoryDataset<ClickThroughBatch>(std::move(data_batches), len),
      BoltDataset(std::move(label_batches), len));
}

}  // namespace thirdai::dataset