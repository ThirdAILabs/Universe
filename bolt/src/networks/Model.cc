#include "Model.h"
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/utils/ProgressBar.h>
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>

namespace thirdai::bolt {

template class Model<bolt::BoltBatch>;
template class Model<dataset::ClickThroughBatch>;

template <typename BATCH_T>
MetricData Model<BATCH_T>::train(
    const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& train_data,
    const dataset::BoltDatasetPtr& train_labels,
    // Clang tidy is disabled for this line because it wants to pass by
    // reference, but shared_ptrs should not be passed by reference
    const LossFunction& loss_fn,  // NOLINT
    float learning_rate, uint32_t epochs, uint32_t rehash, uint32_t rebuild,
    const std::vector<std::string>& metric_names, bool verbose) {
  uint32_t batch_size = train_data->at(0).getBatchSize();
  uint32_t rebuild_batch =
      getRebuildBatch(rebuild, batch_size, train_data->len());
  uint32_t rehash_batch = getRehashBatch(rehash, batch_size, train_data->len());
  uint64_t num_train_batches = train_data->numBatches();

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

      BATCH_T& batch_inputs = train_data->at(batch);

      const BoltBatch& batch_labels = train_labels->at(batch);

      processTrainingBatch(batch_inputs, outputs, batch_labels, loss_fn,
                           learning_rate, rehash_batch, rebuild_batch, metrics);

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
inline void Model<BATCH_T>::processTrainingBatch(
    BATCH_T& batch_inputs, BoltBatch& outputs, const BoltBatch& batch_labels,
    const LossFunction& loss_fn, float learning_rate, uint32_t rehash_batch,
    uint32_t rebuild_batch, MetricAggregator& metrics) {
  if (_batch_iter % 1000 == 999) {
    shuffleRandomNeurons();
  }

#pragma omp parallel for default(none) \
    shared(batch_inputs, batch_labels, outputs, loss_fn, metrics)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    forward(vec_id, batch_inputs, outputs[vec_id], &batch_labels[vec_id]);

    metrics.processSample(outputs[vec_id], batch_labels[vec_id]);

    loss_fn.lossGradients(outputs[vec_id], batch_labels[vec_id],
                          batch_inputs.getBatchSize());

    backpropagate(vec_id, batch_inputs, outputs[vec_id]);
  }

  updateParameters(learning_rate, ++_batch_iter);

  if (_batch_iter % rebuild_batch == (rebuild_batch - 1)) {
    reBuildHashFunctions();
    buildHashTables();
  } else if (_batch_iter % rehash_batch == (rehash_batch - 1)) {
    buildHashTables();
  }
}

template <typename BATCH_T>
BoltBatch Model<BATCH_T>::predict(const BATCH_T& test_data) {
  initializeNetworkState(/* batch_size = */ test_data.getBatchSize(),
                         /* force_dense = */ true);
  BoltBatch outputs = getOutputs(/* batch_size = */ test_data.getBatchSize(),
                                 /* force_dense = */ true);
  MetricAggregator no_metrics(/* metrics = */ {},
                              /* verbose = */ false);
  processTestBatch(
      /* batch_inputs = */ test_data,
      /* outputs = */ outputs,
      /* batch_labels = */ NULL,
      /* output_active_neurons = */ nullptr,
      /* output_activations = */ nullptr,
      /* metrics = */ no_metrics,
      /* compute_metrics = */ false);
  return outputs;
}

template <typename BATCH_T>
InferenceMetricData Model<BATCH_T>::predict(
    const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& test_data,
    const dataset::BoltDatasetPtr& labels, uint32_t* output_active_neurons,
    float* output_activations, const std::vector<std::string>& metric_names,
    bool verbose, uint32_t batch_limit) {
  assert(output_activations != nullptr || output_active_neurons == nullptr);
  bool compute_metrics = labels != nullptr;

  uint32_t batch_size = test_data->at(0).getBatchSize();

  uint64_t num_test_batches = std::min(test_data->numBatches(), batch_limit);

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
  // outputs is only a temporary that gets filled for each batch. We copy
  // into output_activations and output_active_neurons as we go, inside of
  // processTestBatch.
  // TODO(josh): This is a bit hacky, processTestBatch should probably just
  // fill output and we should fill output_activations and output_active_neurons
  // here.
  BoltBatch outputs = getOutputs(batch_size, metrics.forceDenseInference());

  auto test_start = std::chrono::high_resolution_clock::now();
  for (uint32_t batch = 0; batch < num_test_batches; batch++) {
    const BATCH_T& inputs = test_data->at(batch);

    const BoltBatch* batch_labels =
        compute_metrics ? &(*labels)[batch] : nullptr;

    uint32_t* batch_active_neurons =
        output_active_neurons == nullptr
            ? nullptr
            : output_active_neurons +
                  batch * batch_size * getInferenceOutputDim();

    float* batch_activations =
        output_activations == nullptr
            ? nullptr
            : output_activations + batch * batch_size * getInferenceOutputDim();

    processTestBatch(inputs, outputs, batch_labels, batch_active_neurons,
                     batch_activations, metrics, compute_metrics);

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

  auto metric_vals = metrics.getOutputFromInference();

  metric_vals["test_time"] = test_time;

  return metric_vals;
}

template <typename BATCH_T>
inline void Model<BATCH_T>::processTestBatch(const BATCH_T& batch_inputs,
                                             BoltBatch& outputs,
                                             const BoltBatch* batch_labels,
                                             uint32_t* output_active_neurons,
                                             float* output_activations,
                                             MetricAggregator& metrics,
                                             bool compute_metrics) {
  // #pragma omp parallel for default(none)
  //     shared(batch_inputs, batch_labels, outputs, output_active_neurons,
  //            output_activations, metrics, compute_metrics)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    // We set labels to nullptr so that they are not used in sampling during
    // inference.
    forward(vec_id, batch_inputs, outputs[vec_id], /*labels=*/nullptr);

    if (compute_metrics) {
      metrics.processSample(outputs[vec_id], (*batch_labels)[vec_id]);
    }

    if (output_activations != nullptr) {
      assert(outputs[vec_id].len == getInferenceOutputDim());

      const float* start = outputs[vec_id].activations;
      uint32_t offset = vec_id * getInferenceOutputDim();
      std::copy(start, start + outputs[vec_id].len,
                output_activations + offset);
      if (!outputs[vec_id].isDense()) {
        assert(output_active_neurons != nullptr);
        const uint32_t* start = outputs[vec_id].active_neurons;
        std::copy(start, start + outputs[vec_id].len,
                  output_active_neurons + offset);
      }
    }
  }
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