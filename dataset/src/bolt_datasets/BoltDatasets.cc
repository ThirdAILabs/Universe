#include "BoltDatasets.h"
#include "DataLoader.h"
#include "StreamingDataset.h"
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <dataset/src/bolt_datasets/batch_processors/SvmBatchProcessor.h>
#include <dataset/src/parsers/ClickThroughParser.h>
#include <dataset/src/parsers/CsvParser.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <chrono>
#include <fstream>
#include <limits>
#include <memory>

namespace thirdai::dataset {

std::tuple<BoltDatasetPtr, BoltDatasetPtr> loadBoltSvmDataset(
    const std::string& filename, uint32_t batch_size,
    bool softmax_for_multiclass) {
  auto batch_processor =
      std::make_shared<SvmBatchProcessor>(softmax_for_multiclass);

  auto dataset =
      StreamingDataset<bolt::BoltBatch, bolt::BoltBatch>::loadDatasetFromFile(
          filename, batch_size, batch_processor);

  return dataset->loadInMemory();
}

ClickThroughDatasetWithLabels loadClickThroughDataset(
    const std::string& filename, uint32_t batch_size,
    uint32_t num_dense_features, uint32_t num_categorical_features,
    bool sparse_labels) {
  std::cout << "Loading click through dataset from '" << filename << "' ..."
            << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  std::ifstream file = dataset::SafeFileIO::ifstream(filename);

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
      InMemoryDataset<ClickThroughBatch>(std::move(data_batches)),
      BoltDataset(std::move(label_batches)));
}

}  // namespace thirdai::dataset