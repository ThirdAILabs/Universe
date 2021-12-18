#include "DLRM.h"
#include "../../utils/dataset/Dataset.h"
#include <_types/_uint32_t.h>
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

void DLRM::train(const utils::InMemoryDataset<utils::ClickThroughBatch>& train_data, float learning_rate,
                 uint32_t epochs, uint32_t rehash, uint32_t rebuild) {
  utils::InMemoryDataset<utils::ClickThroughBatch> train(
      train_data, batch_size,
      utils::ClickThroughBatchFactory(dense_features, categorical_features));

  utils::InMemoryDataset<utils::ClickThroughBatch> test(
      test_data, batch_size,
      utils::ClickThroughBatchFactory(dense_features, categorical_features));

  uint64_t intermediate_test_batches =
      std::min<uint64_t>(test.numBatches(), max_test_batches);
  uint64_t intermediate_test_vecs =
      std::min<uint64_t>(test.len(), intermediate_test_batches * batch_size);

  // Take max with 1 so that we don't get 0 causing a floating point error.
  uint32_t rebuild_batch = std::max<uint32_t>(rebuild / batch_size, 1);
  uint32_t rehash_batch = std::max<uint32_t>(rehash / batch_size, 1);

  uint64_t num_train_batches = train.numBatches();
  uint32_t print = std::max<uint32_t>(num_train_batches / 10, 1);

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  initializeNetworkForBatchSize(batch_size);

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    std::cout << "---------|" << std::endl;
    auto train_start = std::chrono::high_resolution_clock::now();
    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      if (_iter % 1000 == 999) {
        for (uint32_t i = 0; i < _num_fc_layers; i++) {
          _fc_layers[i]->shuffleRandNeurons();
        }
      }

      processTrainingBatch(train[batch], learning_rate);

      if (_iter % rebuild_batch == (rebuild_batch - 1)) {
        reBuildHashFunctions();
        buildHashTables();
      } else if (_iter % rehash_batch == (rehash_batch - 1)) {
        buildHashTables();
      }

      if ((batch % print) == (print - 1)) {
        std::cout << "." << std::flush;
      }
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

    if (intermediate_test_batches == 0) {
      continue;
    }

    uint32_t correct = 0;
    auto test_start = std::chrono::high_resolution_clock::now();
    for (uint32_t batch = 0; batch < intermediate_test_batches; batch++) {
      correct += processTestBatch(test[batch]);
    }
    auto test_end = std::chrono::high_resolution_clock::now();
    std::cout << "Processed " << intermediate_test_batches
              << " test batches in "
              << std::chrono::duration_cast<std::chrono::seconds>(test_end -
                                                                  test_start)
                     .count()
              << " seconds" << std::endl;

    float accuracy = static_cast<float>(correct) / intermediate_test_vecs;
    _accuracy_per_epoch.push_back(accuracy);
    std::cout << "Accuracy: " << accuracy << " (" << correct << "/"
              << intermediate_test_vecs << ")" << std::endl;
  }

  uint64_t num_test_batches = test.numBatches();
  uint32_t final_correct = 0;
  for (uint32_t batch = 0; batch < num_test_batches; batch++) {
    final_correct += processTestBatch(test[batch]);
  }

  _final_accuracy = static_cast<float>(final_correct) / test.len();
  std::cout << "Accuracy after training: " << _final_accuracy << " ("
            << final_correct << "/" << test.len() << ")" << std::endl;
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

void DLRM::processTrainingBatch(const utils::ClickThroughBatch& batch,
                                float lr) {
#pragma omp parallel for default(none) shared(batch, lr)
  for (uint32_t b = 0; b < batch.getBatchSize(); b++) {
    _embedding_layer->feedForward(b, batch.categoricalFeatures(b).data(),
                                  batch.categoricalFeatures(b).size());

    const utils::DenseVector& vec = batch[b];
    _dense_feature_layer->feedForward(b, /* indices */ nullptr, vec.values,
                                      vec.dim, /* labels */ nullptr,
                                      /* label_len */ 0);

    _fc_layers[0]->feedForward(b, /* indices */ nullptr,
                               _concat_layer_activations[b], _concat_layer_dim,
                               /* labels */ nullptr, /* label_len */ 0);

    uint32_t label = batch.label(b);
    for (uint32_t l = 1; l < _num_fc_layers; l++) {
      FullyConnectedLayer* prev_layer = _fc_layers[l - 1];

      uint32_t* labels = l == _num_fc_layers - 1 ? &label : nullptr;
      uint32_t label_len = l == _num_fc_layers - 1 ? 1 : 0;

      _fc_layers[l]->feedForward(b, prev_layer->getIndices(b),
                                 prev_layer->getValues(b),
                                 prev_layer->getLen(b), labels, label_len);
    }

    _fc_layers[_num_fc_layers - 1]->computeErrors(b, batch.getBatchSize(),
                                                  &label, 1);

    for (uint32_t l = _num_fc_layers - 1; l >= 0; l--) {
      FullyConnectedLayer* prev_layer = _fc_layers[l - 1];

      _fc_layers[l]->backpropagate(
          b, prev_layer->getIndices(b), prev_layer->getValues(b),
          prev_layer->getErrors(b), prev_layer->getLen(b));
    }

    _dense_feature_layer->backpropagateFirstLayer(b, /* indices */ nullptr,
                                                  vec.values, vec.dim);

    _embedding_layer->backpropagate(b);
  }

  ++_iter;
  _embedding_layer->updateParameters(lr, _iter, BETA1, BETA2, EPS);
  _dense_feature_layer->updateParameters(lr, _iter, BETA1, BETA2, EPS);
  for (uint32_t l = 0; l < _num_fc_layers; l++) {
    _fc_layers[l]->updateParameters(lr, _iter, BETA1, BETA2, EPS);
  }
}

uint32_t DLRM::processTestBatch(const utils::ClickThroughBatch& batch) {
  for (uint32_t l = 0; l < _num_fc_layers; l++) {
    _fc_layers[l]->setSparsity(1.0);
  }

  std::atomic<uint32_t> tp = 0;
  std::atomic<uint32_t> fp = 0;
  std::atomic<uint32_t> tn = 0;
  std::atomic<uint32_t> fn = 0;

#pragma omp parallel for default(none) shared(batch, tp, fp, tn, fn)
  for (uint32_t b = 0; b < batch.getBatchSize(); b++) {
    _embedding_layer->feedForward(b, batch.categoricalFeatures(b).data(),
                                  batch.categoricalFeatures(b).size());

    const utils::DenseVector& vec = batch[b];
    _dense_feature_layer->feedForward(b, /* indices */ nullptr, vec.values,
                                      vec.dim, /* labels */ nullptr,
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
    uint32_t score = activations[1] > activations[0] ? 2 : 0;
    if (batch.label(b)) {
      ++score;
    }

    switch (score) {
      case 0:
        ++tn;
        break;
      case 1:
        ++fn;
        break;
      case 2:
        ++fp;
        break;
      case 3:
        ++fn;
        break;
    }
  }

  for (uint32_t l = 0; l < _num_fc_layers; l++) {
    _fc_layers[l]->setSparsity(_fc_layer_configs[l].sparsity);
  }

  return 0;
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