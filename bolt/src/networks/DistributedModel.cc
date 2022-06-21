#include "DistributedModel.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/utils/ProgressBar.h>
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace thirdai::bolt {

template class DistributedModel<bolt::BoltBatch>;
template class DistributedModel<dataset::ClickThroughBatch>;

template <typename BATCH_T>
uint32_t DistributedModel<BATCH_T>::initTrainDistributed(
    std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& train_data,
    const dataset::BoltDatasetPtr& train_labels,
    // Clang tidy is disabled for this line because it wants to pass by
    // reference, but shared_ptrs should not be passed by reference
    uint32_t rehash, uint32_t rebuild, bool verbose) {
  _train_data = train_data;
  _train_labels = train_labels;
  uint32_t batch_size = _train_data->at(0).getBatchSize();
  _rebuild_batch =
      getRebuildBatchDistributed(rebuild, batch_size, train_data->len());
  _rehash_batch =
      getRehashBatchDistributed(rehash, batch_size, train_data->len());

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  initializeNetworkState(batch_size, false);
  _outputs = getOutputs(batch_size, false);

  if (verbose) {
    std::cout << "Distributed Network initialization done on this Node"
              << std::endl;
  }
  return train_data->numBatches();
}

template <typename BATCH_T>
void DistributedModel<BATCH_T>::calculateGradientDistributed(
    uint32_t batch, const LossFunction& loss_fn) {
  BATCH_T& batch_inputs = _train_data->at(batch);

  const BoltBatch& batch_labels = _train_labels->at(batch);

  if (_batch_iter % 1000 == 999) {
    shuffleRandomNeurons();
  }

#pragma omp parallel for default(none) \
    shared(batch_inputs, batch_labels, _outputs, loss_fn)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    forward(vec_id, batch_inputs, _outputs[vec_id], &batch_labels[vec_id]);

    loss_fn.lossGradients(_outputs[vec_id], batch_labels[vec_id],
                          batch_inputs.getBatchSize());
    backpropagate(vec_id, batch_inputs, _outputs[vec_id]);
  }
}

template <typename BATCH_T>
void DistributedModel<BATCH_T>::updateParametersDistributed(
    float learning_rate) {
  updateParameters(learning_rate, ++_batch_iter);
  if (_batch_iter % _rebuild_batch == (_rebuild_batch - 1)) {
    reBuildHashFunctions();
    buildHashTables();
  } else if (_batch_iter % _rehash_batch == (_rehash_batch - 1)) {
    buildHashTables();
  }
}

template <typename BATCH_T>
InferenceMetricData DistributedModel<BATCH_T>::predictDistributed(
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

    processTestBatchDistributed(inputs, outputs, batch_labels,
                                batch_active_neurons, batch_activations,
                                metrics, compute_metrics);

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
InferenceMetricData DistributedModel<BATCH_T>::predictOnStreamDistributed(
    const std::shared_ptr<dataset::StreamingDataset<BATCH_T>>& test_data,
    const std::vector<std::string>& metric_names,
    std::optional<std::function<void(const bolt::BoltBatch&, uint32_t)>>
        batch_callback,
    bool verbose) {
  MetricAggregator metrics(metric_names, verbose);

  uint32_t batch_size = test_data->getMaxBatchSize();
  initializeNetworkState(batch_size, metrics.forceDenseInference());
  BoltBatch outputs = getOutputs(batch_size, metrics.forceDenseInference());

  if (verbose) {
    std::cout << std::endl
              << "Processing test streaming dataset..." << std::endl;
  }

  auto test_start = std::chrono::high_resolution_clock::now();

  while (auto batch = test_data->nextBatch()) {
    processTestBatchDistributed(batch->first, outputs, &batch->second,
                                /* output_active_neurons=*/nullptr,
                                /* output_activations=*/nullptr, metrics,
                                /* compute_metrics= */ true);

    if (batch_callback) {
      batch_callback.value()(outputs, batch->first.getBatchSize());
    }
  }

  auto test_end = std::chrono::high_resolution_clock::now();

  int64_t test_time =
      std::chrono::duration_cast<std::chrono::seconds>(test_end - test_start)
          .count();

  if (verbose) {
    std::cout << "Processed streaming dataset in " << test_time << " seconds."
              << std::endl;
  }

  metrics.logAndReset();
  auto metric_data = metrics.getOutputFromInference();
  metric_data["epoch_times"] = static_cast<double>(test_time);

  return metric_data;
}

template <typename BATCH_T>
inline void DistributedModel<BATCH_T>::processTestBatchDistributed(
    const BATCH_T& batch_inputs, BoltBatch& outputs,
    const BoltBatch* batch_labels, uint32_t* output_active_neurons,
    float* output_activations, MetricAggregator& metrics,
    bool compute_metrics) {
#pragma omp parallel for default(none)                                 \
    shared(batch_inputs, batch_labels, outputs, output_active_neurons, \
           output_activations, metrics, compute_metrics)
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
uint32_t DistributedModel<BATCH_T>::getRehashBatchDistributed(
    uint32_t rehash, uint32_t batch_size, uint32_t data_len) {
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
uint32_t DistributedModel<BATCH_T>::getRebuildBatchDistributed(
    uint32_t rebuild, uint32_t batch_size, uint32_t data_len) {
  rebuild = rebuild != 0 ? rebuild : (data_len / 4);
  return std::max<uint32_t>(rebuild / batch_size, 1);
}

// The following functions are added to make Bolt Distributed work.
// These functions are going to be extended to python with the help
// of pybindings.

}  // namespace thirdai::bolt