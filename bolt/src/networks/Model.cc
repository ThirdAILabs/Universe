#include "Model.h"
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/utils/ProgressBar.h>
#include <dataset/src/batch_types/BoltInputBatch.h>
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>

namespace thirdai::bolt {

template class Model<dataset::BoltInputBatch>;
template class Model<dataset::ClickThroughBatch>;

template <typename BATCH_T>
MetricData Model<BATCH_T>::train(
    dataset::InMemoryDataset<BATCH_T>& train_data,
    // Clang tidy is disabled for this line because it wants to pass by
    // reference, but shared_ptrs should not be passed by reference
    const LossFunction& loss_fn,  // NOLINT
    float learning_rate, uint32_t epochs, uint32_t rehash, uint32_t rebuild,
    const std::vector<std::string>& metric_names, bool verbose) {
  uint32_t batch_size = train_data[0].getBatchSize();
  uint32_t rebuild_batch =
      getRebuildBatch(rebuild, batch_size, train_data.len());
  uint32_t rehash_batch = getRehashBatch(rehash, batch_size, train_data.len());
  uint64_t num_train_batches = train_data.numBatches();

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  initializeNetworkState(batch_size, false);
  BoltBatch outputs = getOutputs(batch_size, false);

  std::vector<double> time_per_epoch;

  MetricAggregator metrics(metric_names, verbose);

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    if (verbose) {
      std::cout << "\nEpoch " << (_epoch_count + 1) << ':' << std::endl;
    }
    ProgressBar bar(num_train_batches, verbose);
    auto train_start = std::chrono::high_resolution_clock::now();

    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      if (_batch_iter % 1000 == 999) {
        shuffleRandomNeurons();
      }

      BATCH_T& inputs = train_data[batch];

#pragma omp parallel for default(none) shared(inputs, outputs, loss_fn, metrics)
      for (uint32_t i = 0; i < inputs.getBatchSize(); i++) {
        forward(i, inputs, outputs[i], /* train=*/true);

        loss_fn.lossGradients(outputs[i], inputs.labels(i),
                              inputs.getBatchSize());

        backpropagate(i, inputs, outputs[i]);

        metrics.processSample(outputs[i], inputs.labels(i));
      }

      updateParameters(learning_rate, ++_batch_iter);

      if (_batch_iter % rebuild_batch == (rebuild_batch - 1)) {
        reBuildHashFunctions();
        buildHashTables();
      } else if (_batch_iter % rehash_batch == (rehash_batch - 1)) {
        buildHashTables();
      }

      bar.increment();
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    int64_t epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
                             train_end - train_start)
                             .count();

    time_per_epoch.push_back(static_cast<double>(epoch_time));
    if (verbose) {
      std::cout << std::endl
                << "Processed " << num_train_batches << " training batches in "
                << epoch_time << " seconds" << std::endl;
    }
    _epoch_count++;

    metrics.logAndReset();
  }

  auto metric_data = metrics.getOutput();
  metric_data["epoch_times"] = std::move(time_per_epoch);

  return metric_data;
}

template <typename BATCH_T>
MetricData Model<BATCH_T>::predict(
    const dataset::InMemoryDataset<BATCH_T>& test_data,
    float* output_activations, const std::vector<std::string>& metric_names,
    bool verbose, uint32_t batch_limit) {
  uint32_t batch_size = test_data[0].getBatchSize();

  uint64_t num_test_batches = std::min(test_data.numBatches(), batch_limit);

  MetricAggregator metrics(metric_names, verbose);

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  // If sparse inference is not enabled we want the outptus to be dense,
  // otherwise we want whatever the default for the layer is.
  initializeNetworkState(batch_size, metrics.forceDenseInference());

  ProgressBar bar(num_test_batches, verbose);

  // Don't force dense inference if the metric does not allow it.
  // This is not the same as enable_sparse_inference(), which also freezes hash
  // tables.
  BoltBatch outputs = getOutputs(batch_size, metrics.forceDenseInference());

  auto test_start = std::chrono::high_resolution_clock::now();
  for (uint32_t batch = 0; batch < num_test_batches; batch++) {
    const BATCH_T& inputs = test_data[batch];

#pragma omp parallel for default(none) \
    shared(inputs, outputs, output_activations, metrics, batch, batch_size)
    for (uint32_t i = 0; i < inputs.getBatchSize(); i++) {
      forward(i, inputs, outputs[i], /* train=*/false);

      metrics.processSample(outputs[i], inputs.labels(i));

      if (output_activations != nullptr && outputs[i].isDense()) {
        const float* start = outputs[i].activations;
        std::copy(start, start + outputs[i].len,
                  output_activations + (batch * batch_size + i) * outputDim());
      }
    }

    bar.increment();
  }

  auto test_end = std::chrono::high_resolution_clock::now();
  int64_t test_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          test_end - test_start)
                          .count();
  if (verbose) {
    std::cout << std::endl
              << "Processed " << num_test_batches << " test batches in "
              << test_time << " milliseconds" << std::endl;
  }

  metrics.logAndReset();

  auto metric_vals = metrics.getOutput();

  metric_vals["test_time"].push_back(test_time);

  return metric_vals;
}

static constexpr uint32_t RehashAutoTuneThreshold = 100000;
static constexpr uint32_t RehashAutoTuneFactor1 = 100;
static constexpr uint32_t RehashAutoTuneFactor2 = 20;

template <typename BATCH_T>
uint32_t Model<BATCH_T>::getRehashBatch(uint32_t rehash, uint32_t batch_size,
                                        uint32_t data_len) {
  if (rehash == 0) {
    if (data_len < RehashAutoTuneThreshold) {
      rehash = data_len / RehashAutoTuneFactor2;
    } else {
      rehash = data_len / RehashAutoTuneFactor1;
    }
  }
  return std::max<uint32_t>(rehash / batch_size, 1);
}

template <typename BATCH_T>
uint32_t Model<BATCH_T>::getRebuildBatch(uint32_t rebuild, uint32_t batch_size,
                                         uint32_t data_len) {
  rebuild = rebuild != 0 ? rebuild : (data_len / 4);
  return std::max<uint32_t>(rebuild / batch_size, 1);
}

}  // namespace thirdai::bolt