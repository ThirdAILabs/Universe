#include "DLRM.h"
#include <bolt/utils/ProgressBar.h>
#include <atomic>
#include <chrono>
#include <stdexcept>

namespace thirdai::bolt {

struct PrecisionRecal {
  uint32_t true_positive, false_positive;
  uint32_t true_negative, false_negative;
};

DLRM::DLRM(EmbeddingLayerConfig embedding_config,
           FullyConnectedLayerConfig dense_feature_layer_config,
           std::vector<FullyConnectedLayerConfig> fc_layer_configs,
           uint32_t input_dim)
    : _fc_layer_configs(std::move(fc_layer_configs)), _iter(0) {
  _embedding_layer = new EmbeddingLayer(embedding_config);

  if (dense_feature_layer_config.sparsity != 1.0) {
    throw std::invalid_argument(
        "Dense feature layer must be have sparsity 1.0");
  }
  _dense_feature_layer =
      new FullyConnectedLayer(dense_feature_layer_config, input_dim);

  _num_fc_layers = _fc_layer_configs.size();
  _fc_layers = new FullyConnectedLayer*[_num_fc_layers];

  _concat_layer_dim =
      _embedding_layer->getEmbeddingDim() + dense_feature_layer_config.dim;

  for (uint32_t l = 0; l < _fc_layer_configs.size(); l++) {
    uint32_t prev_dim =
        l > 0 ? _fc_layer_configs.at(l - 1).dim : _concat_layer_dim;
    _fc_layers[l] = new FullyConnectedLayer(_fc_layer_configs.at(l), prev_dim);
  }
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
  initializeNetworkForBatchSize(batch_size);

  ProgressBar bar(num_train_batches);

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    bar.reset();
    auto train_start = std::chrono::high_resolution_clock::now();
    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      if (_iter % 1000 == 999) {
        for (uint32_t i = 0; i < _num_fc_layers; i++) {
          _fc_layers[i]->shuffleRandNeurons();
        }
      }

      processTrainingBatch(train_data[batch], learning_rate);

      if (_iter % rebuild_batch == (rebuild_batch - 1)) {
        reBuildHashFunctions();
        buildHashTables();
      } else if (_iter % rehash_batch == (rehash_batch - 1)) {
        buildHashTables();
      }

      bar.update();
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    int64_t epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
                             train_end - train_start)
                             .count();
    _time_per_epoch.push_back(epoch_time);
    std::cout << std::endl
              << "Epoch: " << epoch << "\nProcessed " << num_train_batches
              << " training batches in " << epoch_time << " seconds"
              << std::endl;
  }
}

void DLRM::testImpl(
    const dataset::InMemoryDataset<dataset::ClickThroughBatch>& test_data,
    float* scores) {
  uint32_t batch_size = test_data.at(0).getBatchSize();
  uint64_t num_test_batches = test_data.numBatches();

  initializeNetworkForBatchSize(batch_size);

  ProgressBar bar(num_test_batches);

  uint32_t offset = 0;
  for (const auto& batch : test_data) {
    processTestBatch(batch, scores + offset);
    offset += batch.getBatchSize();
    bar.update();
  }
}

void DLRM::initializeNetworkForBatchSize(uint32_t batch_size) {
  _concat_layer_activations = new float*[batch_size];
  _concat_layer_errors = new float*[batch_size];

  float** embedding_layer_activations = new float*[batch_size];
  float** embedding_layer_errors = new float*[batch_size];
  float** dense_feature_layer_activations = new float*[batch_size];
  float** dense_feature_layer_errors = new float*[batch_size];

  uint32_t embedding_dim = _embedding_layer->getEmbeddingDim();

  for (uint32_t i = 0; i < batch_size; i++) {
    _concat_layer_activations[i] = new float[_concat_layer_dim]();
    _concat_layer_errors[i] = new float[_concat_layer_dim]();

    embedding_layer_activations[i] = _concat_layer_activations[i];
    embedding_layer_errors[i] = _concat_layer_errors[i];

    dense_feature_layer_activations[i] =
        _concat_layer_activations[i] + embedding_dim;
    dense_feature_layer_errors[i] = _concat_layer_errors[i] + embedding_dim;
  }

  _embedding_layer->initializeLayer(batch_size, embedding_layer_activations,
                                    embedding_layer_errors);
  _dense_feature_layer->initializeLayer(
      batch_size, dense_feature_layer_activations, dense_feature_layer_errors);

  for (uint32_t l = 0; l < _num_fc_layers; l++) {
    _fc_layers[l]->initializeLayer(batch_size);
  }
}

void DLRM::processTrainingBatch(const dataset::ClickThroughBatch& batch,
                                float lr) {
#pragma omp parallel for default(none) shared(batch, lr)
  for (uint32_t b = 0; b < batch.getBatchSize(); b++) {
    _embedding_layer->feedForward(b, batch.categoricalFeatures(b).data(),
                                  batch.categoricalFeatures(b).size());

    const dataset::DenseVector& vec = batch[b];
    _dense_feature_layer->feedForward(b, /* indices */ nullptr, vec._values,
                                      vec.dim(), /* labels */ nullptr,
                                      /* label_len */ 0);

    _fc_layers[0]->feedForward(b, /* indices */ nullptr,
                               _concat_layer_activations[b], _concat_layer_dim,
                               /* labels */ nullptr, /* label_len */ 0);

    uint32_t label = batch.label(b);
    float label_val = 1.0;
    for (uint32_t l = 1; l < _num_fc_layers; l++) {
      FullyConnectedLayer* prev_layer = _fc_layers[l - 1];

      uint32_t* labels = l == _num_fc_layers - 1 ? &label : nullptr;
      uint32_t label_len = l == _num_fc_layers - 1 ? 1 : 0;

      _fc_layers[l]->feedForward(b, prev_layer->getIndices(b),
                                 prev_layer->getValues(b),
                                 prev_layer->getLen(b), labels, label_len);
    }

    _fc_layers[_num_fc_layers - 1]->computeMeanSquaredErrors(
        b, batch.getBatchSize(), &label, &label_val, 1);

    for (uint32_t l = _num_fc_layers; l > 0; l--) {
      uint32_t layer = l -1;
      FullyConnectedLayer* prev_layer = _fc_layers[layer - 1];

      _fc_layers[layer]->backpropagate(
          b, prev_layer->getIndices(b), prev_layer->getValues(b),
          prev_layer->getErrors(b), prev_layer->getLen(b));
    }

    _dense_feature_layer->backpropagateFirstLayer(b, /* indices */ nullptr,
                                                  vec._values, vec.dim());

    _embedding_layer->backpropagate(b);
  }

  ++_iter;
  _embedding_layer->updateParameters(lr, _iter, BETA1, BETA2, EPS);
  _dense_feature_layer->updateParameters(lr, _iter, BETA1, BETA2, EPS);
  for (uint32_t l = 0; l < _num_fc_layers; l++) {
    _fc_layers[l]->updateParameters(lr, _iter, BETA1, BETA2, EPS);
  }
}

void DLRM::processTestBatch(const dataset::ClickThroughBatch& batch,
                            float* scores) {
  for (uint32_t l = 0; l < _num_fc_layers; l++) {
    _fc_layers[l]->setSparsity(1.0);
  }

#pragma omp parallel for default(none) shared(batch, scores)
  for (uint32_t b = 0; b < batch.getBatchSize(); b++) {
    _embedding_layer->feedForward(b, batch.categoricalFeatures(b).data(),
                                  batch.categoricalFeatures(b).size());

    const dataset::DenseVector& vec = batch[b];
    _dense_feature_layer->feedForward(b, /* indices */ nullptr, vec._values,
                                      vec.dim(), /* labels */ nullptr,
                                      /* label_len */ 0);

    _fc_layers[0]->feedForward(b, /* indices */ nullptr,
                               _concat_layer_activations[b], _concat_layer_dim,
                               /* labels */ nullptr, /* label_len */ 0);

    for (uint32_t l = 1; l < _num_fc_layers; l++) {
      FullyConnectedLayer* prev_layer = _fc_layers[l - 1];

      _fc_layers[l]->feedForward(b, prev_layer->getIndices(b),
                                 prev_layer->getValues(b),
                                 prev_layer->getLen(b), nullptr, 0);
    }

    const float* activations = _fc_layers[_num_fc_layers - 1]->getValues(b);
    scores[b] = activations[0];
  }

  for (uint32_t l = 0; l < _num_fc_layers; l++) {
    _fc_layers[l]->setSparsity(_fc_layer_configs[l].sparsity);
  }
}

void DLRM::reBuildHashFunctions() {
  for (uint32_t l = 0; l < _num_fc_layers; l++) {
    _fc_layers[l]->reBuildHashFunction();
  }
}

void DLRM::buildHashTables() {
  for (uint32_t l = 0; l < _num_fc_layers; l++) {
    _fc_layers[l]->buildHashTables();
  }
}

}  // namespace thirdai::bolt