#include "BatchProcessor.h"
#include "Datasets.h"
#include "InMemoryDataset.h"
#include "StreamingDataset.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/batch_processors/SvmBatchProcessor.h>
#include <dataset/src/batch_types/BoltTokenBatch.h>
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <dataset/src/parsers/ClickThroughParser.h>

namespace thirdai::dataset {

struct SvmDatasetLoader {
  static std::tuple<BoltDatasetPtr, BoltDatasetPtr> loadDataset(
      const std::string& filename, uint32_t batch_size,
      bool softmax_for_multiclass = true) {
    auto batch_processor =
        std::make_shared<SvmBatchProcessor>(softmax_for_multiclass);

    auto dataset =
        StreamingDataset<bolt::BoltBatch, bolt::BoltBatch>::loadDatasetFromFile(
            filename, batch_size, batch_processor);

    return dataset->loadInMemory();
  }
};

// Everything below this comment should be deleted when the old bolt API is
// depreciated.

using ClickThroughDataset = InMemoryDataset<ClickThroughBatch>;
using ClickThroughDatasetPtr = std::shared_ptr<ClickThroughDataset>;

class ClickThroughDatasetWithLabels {
 public:
  ClickThroughDatasetPtr data;
  BoltDatasetPtr labels;

  explicit ClickThroughDatasetWithLabels(
      InMemoryDataset<ClickThroughBatch>&& _data, BoltDataset&& _labels)
      : data(std::make_shared<InMemoryDataset<ClickThroughBatch>>(
            std::move(_data))),
        labels(std::make_shared<BoltDataset>(std::move(_labels))) {}
};

struct ClickThroughDatasetLoader {
  static ClickThroughDatasetWithLabels loadDataset(
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
};

}  // namespace thirdai::dataset