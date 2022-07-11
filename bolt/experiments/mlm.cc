#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <dataset/src/bolt_datasets/DataLoader.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <dataset/src/bolt_datasets/batch_processors/MaskedSentenceBatchProcessor.h>

using namespace thirdai::bolt;  // NOLINT

auto loadMLMDataset(const std::string& filename, uint32_t batch_size,
                    uint32_t pairgram_range) {
  auto data_loader = std::make_shared<thirdai::dataset::SimpleFileDataLoader>(
      filename, batch_size);

  auto batch_processor =
      std::make_shared<thirdai::dataset::MaskedSentenceBatchProcessor>(
          pairgram_range);

  auto dataset = std::make_shared<thirdai::dataset::StreamingDataset<
      thirdai::dataset::MaskedSentenceBatch>>(data_loader, batch_processor);

  return dataset->loadInMemory();
}

int main() {
  std::string train_file =
      "/share/data/BERT/sentences_tokenized_shuffled_trimmed_10M.txt";
  std::string test_file =
      "/share/data/BERT/sentences_tokenized_shuffled_trimmed_test_100k.txt";

  auto input = std::make_shared<Input>(100000);
  auto hidden =
      std::make_shared<FullyConnectedNode>(200, "relu")->addPredecessor(input);
  auto output = std::make_shared<FullyConnectedNode>(30224, 0.01, "softmax")
                    ->addPredecessor(hidden);

  BoltGraph model({input}, output);
  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  auto [train_x, train_y] = loadMLMDataset(train_file, 2048, 100000);
  auto [test_x, test_y] = loadMLMDataset(test_file, 2048, 100000);

  auto train_cfg = TrainConfig::makeConfig(0.0001, 1)
                       .withMetrics({"mean_squared_error"})
                       .withRebuildHashTables(10000)
                       .withReconstructHashFunctions(100000);

  auto predict_cfg =
      PredictConfig::makeConfig().withMetrics({"categorical_accuracy"});

  for (uint32_t e = 0; e < 2; e++) {
    model.train(train_x, train_y, train_cfg);
    model.predict(test_x, test_y, predict_cfg);
  }

  return 0;
}