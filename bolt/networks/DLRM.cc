#include "DLRM.h"
#include <bolt/layers/LossFunctions.h>
#include <bolt/utils/ProgressBar.h>
#include <atomic>
#include <chrono>
#include <stdexcept>

namespace thirdai::bolt {

DLRM::DLRM(EmbeddingLayerConfig embedding_config,
           std::vector<FullyConnectedLayerConfig> bottom_mlp_configs,
           std::vector<FullyConnectedLayerConfig> top_mlp_configs,
           uint32_t dense_feature_dim)
    : _embedding_layer(embedding_config),
      _bottom_mlp(bottom_mlp_configs, dense_feature_dim),
      _top_mlp(top_mlp_configs, (embedding_config.lookup_size *
                                 embedding_config.num_embedding_lookups) +
                                    bottom_mlp_configs.back().dim),

      _iter(0),
      _epoch_count(0) {
  if (bottom_mlp_configs.back().sparsity != 1.0) {
    throw std::invalid_argument(
        "Dense feature layer must be have sparsity 1.0");
  }

  if (top_mlp_configs.back().dim != 1) {
    throw std::invalid_argument("Output layer must have dimension 1");
  }

  _concat_layer_dim =
      _embedding_layer.getEmbeddingDim() + bottom_mlp_configs.back().dim;
}

void DLRM::train(
    const dataset::InMemoryDataset<dataset::ClickThroughBatch>& train_data,
    float learning_rate, uint32_t epochs, uint32_t rehash, uint32_t rebuild) {
  uint32_t batch_size = train_data.at(0).getBatchSize();
  // Take max with 1 so that we don't get 0 causing a floating point error.
  uint32_t rebuild_batch = std::max<uint32_t>(rebuild / batch_size, 1);
  uint32_t rehash_batch = std::max<uint32_t>(rehash / batch_size, 1);

  uint64_t num_train_batches = train_data.numBatches();

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  initializeNetworkForBatchSize(batch_size, false);

  BatchState output(1, batch_size, true);

  MeanSquaredError MSE;

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    std::cout << "\nEpoch " << (_epoch_count + 1) << ':' << std::endl;
    ProgressBar bar(num_train_batches);

    auto train_start = std::chrono::high_resolution_clock::now();
    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      if (_iter % 1000 == 999) {
        _bottom_mlp.shuffleRandomNeurons();
        _top_mlp.shuffleRandomNeurons();
      }

      const dataset::ClickThroughBatch& input_batch = train_data[batch];

#pragma omp parallel for default(none) \
    shared(input_batch, output, batch_size, MSE)
      for (uint32_t b = 0; b < input_batch.getBatchSize(); b++) {
        VectorState dense_input = VectorState::makeDenseInputState(
            input_batch[b]._values, input_batch[b].dim());

        forward(b, dense_input, input_batch.categoricalFeatures(b), output[b]);

        float label = static_cast<float>(input_batch.label(b));

        MSE(output[b], batch_size, nullptr, &label, 1);

        backpropagate(b, dense_input, output[b]);
      }

      _bottom_mlp.updateParameters(learning_rate);
      _embedding_layer.updateParameters(learning_rate, ++_iter, BETA1, BETA2,
                                        EPS);
      _top_mlp.updateParameters(learning_rate);

      if (_iter % rebuild_batch == (rebuild_batch - 1)) {
        reBuildHashFunctions();
        buildHashTables();
      } else if (_iter % rehash_batch == (rehash_batch - 1)) {
        buildHashTables();
      }

      bar.increment();
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    int64_t epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
                             train_end - train_start)
                             .count();
    std::cout << std::endl
              << "Processed " << num_train_batches << " training batches in "
              << epoch_time << " seconds" << std::endl;

    _epoch_count++;
  }
}

void DLRM::testImpl(
    const dataset::InMemoryDataset<dataset::ClickThroughBatch>& test_data,
    float* scores) {
  uint32_t batch_size = test_data.at(0).getBatchSize();
  uint64_t num_test_batches = test_data.numBatches();

  initializeNetworkForBatchSize(batch_size, true);

  BatchState output(1, batch_size, true);

  ProgressBar bar(num_test_batches);

  uint32_t cnt = 0;

  auto test_start = std::chrono::high_resolution_clock::now();

  for (const auto& batch : test_data) {
#pragma omp parallel for default(none) shared(batch, output, scores, cnt)
    for (uint32_t b = 0; b < batch.getBatchSize(); b++) {
      VectorState dense_input =
          VectorState::makeDenseInputState(batch[b]._values, batch[b].dim());

      forward(b, dense_input, batch.categoricalFeatures(b), output[b]);

      scores[cnt + b] = output[b].activations[0];
    }

    cnt += batch.getBatchSize();

    bar.increment();
  }

  auto test_end = std::chrono::high_resolution_clock::now();
  int64_t test_time =
      std::chrono::duration_cast<std::chrono::seconds>(test_end - test_start)
          .count();
  std::cout << std::endl
            << "Processed " << num_test_batches << " test batches in "
            << test_time << " seconds" << std::endl;
}

void DLRM::initializeNetworkForBatchSize(uint32_t batch_size,
                                         bool force_dense) {
  _concat_layer_state = BatchState(_concat_layer_dim, batch_size, true);

  uint32_t embedding_dim = _embedding_layer.getEmbeddingDim();
  uint32_t bottom_mlp_output_dim = _bottom_mlp.getLayerSizes().back();

  _bottom_mlp_output.clear();
  _embedding_layer_output.clear();

  for (uint32_t b = 0; b < batch_size; b++) {
    const VectorState& concat_vec = _concat_layer_state[b];

    _bottom_mlp_output.push_back(VectorState::makeDenseState(
        concat_vec.activations, concat_vec.gradients, bottom_mlp_output_dim));

    _embedding_layer_output.push_back(VectorState::makeDenseState(
        concat_vec.activations + bottom_mlp_output_dim,
        concat_vec.gradients + bottom_mlp_output_dim, embedding_dim));
  }

  _embedding_layer.initializeLayer(batch_size);

  _bottom_mlp.createBatchStates(batch_size, force_dense);
  _top_mlp.createBatchStates(batch_size, force_dense);
}

void DLRM::forward(uint32_t batch_index, const VectorState& dense_input,
                   const std::vector<uint32_t>& categorical_features,
                   VectorState& output) {
  _bottom_mlp.forward(batch_index, dense_input, _bottom_mlp_output[batch_index],
                      nullptr, 0);

  _embedding_layer.forward(batch_index, categorical_features,
                           _embedding_layer_output[batch_index]);

  _top_mlp.forward(batch_index, _concat_layer_state[batch_index], output,
                   nullptr, 0);
}

void DLRM::backpropagate(uint32_t batch_index, VectorState& dense_input,
                         VectorState& output) {
  _top_mlp.backpropagate<false>(batch_index, _concat_layer_state[batch_index],
                                output);

  _embedding_layer.backpropagate(batch_index,
                                 _embedding_layer_output[batch_index]);

  _bottom_mlp.backpropagate<true>(batch_index, dense_input,
                                  _bottom_mlp_output[batch_index]);
}

void DLRM::reBuildHashFunctions() {
  _bottom_mlp.reBuildHashFunctions();
  _top_mlp.reBuildHashFunctions();
}

void DLRM::buildHashTables() {
  _bottom_mlp.buildHashTables();
  _top_mlp.buildHashTables();
}

}  // namespace thirdai::bolt