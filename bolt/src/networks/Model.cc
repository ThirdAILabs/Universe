#include "Model.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/utils/ProgressBar.h>
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace thirdai::bolt {

template class Model<bolt::BoltBatch>;
template class Model<dataset::ClickThroughBatch>;

template <typename BATCH_T>
MetricData Model<BATCH_T>::train(
    std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& train_data,
    const dataset::BoltDatasetPtr& train_labels,
    // Clang tidy is disabled for this line because it wants to pass by
    // reference, but shared_ptrs should not be passed by reference
    const LossFunction& loss_fn, float learning_rate, uint32_t epochs,
    uint32_t rehash, uint32_t rebuild,
    const std::vector<std::string>& metric_names, bool verbose) {
  uint32_t batch_size = train_data->at(0).getBatchSize();
  uint32_t rebuild_batch =
      getRebuildBatch(rebuild, batch_size, train_data->len());
  uint32_t rehash_batch = getRehashBatch(rehash, batch_size, train_data->len());
  uint64_t num_train_batches = train_data->numBatches();

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  initializeNetworkState(batch_size, /* use_sparsity= */ true);
  BoltBatch outputs = getOutputs(batch_size, /* use_sparsity= */ true);

  std::vector<double> time_per_epoch;

  MetricAggregator metrics(metric_names, verbose);

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    if (verbose) {
      std::cout << "\nEpoch " << (_epoch_count + 1) << ':' << std::endl;
    }
    ProgressBar bar(num_train_batches, verbose);
    auto train_start = std::chrono::high_resolution_clock::now();

    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      BATCH_T& batch_inputs = train_data->at(batch);

      const BoltBatch& batch_labels = train_labels->at(batch);

      processTrainingBatch(batch_inputs, outputs, batch_labels, loss_fn,
                           learning_rate, metrics);

      updateSampling(/* rehash_batch */ rehash_batch,
                     /* rebuild_batch= */ rebuild_batch);

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
MetricData Model<BATCH_T>::trainOnStream(
    std::shared_ptr<dataset::StreamingDataset<BATCH_T, BoltBatch>>& train_data,
    const LossFunction& loss_fn, float learning_rate, uint32_t rehash_batch,
    uint32_t rebuild_batch, const std::vector<std::string>& metric_names,
    uint32_t metric_log_batch_interval, bool verbose) {
  MetricAggregator metrics(metric_names, verbose);

  uint32_t batch_size = train_data->getMaxBatchSize();
  initializeNetworkState(batch_size, /* use_sparsity= */ true);
  BoltBatch outputs = getOutputs(batch_size, /* use_sparsity= */ true);

  if (verbose) {
    std::cout << std::endl
              << "Processing training streaming dataset..." << std::endl;
  }

  auto train_start = std::chrono::high_resolution_clock::now();

  uint32_t batch_count = 0;
  while (auto batch = train_data->nextBatchTuple()) {
    processTrainingBatch(
        /* batch_inputs=*/std::get<0>(batch.value()), /* outputs= */ outputs,
        /* batch_labels= */ std::get<1>(batch.value()), /* loss_fn= */ loss_fn,
        /* learning_rate= */ learning_rate, /* metrics= */ metrics);

    updateSampling(/* rehash_batch */ rehash_batch,
                   /* rebuild_batch= */ rebuild_batch);

    batch_count++;
    if (batch_count == metric_log_batch_interval) {
      metrics.logAndReset();
      batch_count = 0;
    }
  }

  auto train_end = std::chrono::high_resolution_clock::now();

  int64_t train_time =
      std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start)
          .count();

  if (verbose) {
    std::cout << "Processed streaming dataset in " << train_time << " seconds."
              << std::endl;
  }

  metrics.logAndReset();
  auto metric_data = metrics.getOutput();
  metric_data["epoch_times"] = {static_cast<double>(train_time)};

  return metric_data;
}

template <typename BATCH_T>
inline void Model<BATCH_T>::processTrainingBatch(BATCH_T& batch_inputs,
                                                 BoltBatch& outputs,
                                                 const BoltBatch& batch_labels,
                                                 const LossFunction& loss_fn,
                                                 float learning_rate,
                                                 MetricAggregator& metrics) {
#pragma omp parallel for default(none) \
    shared(batch_inputs, batch_labels, outputs, loss_fn, metrics)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    forward(vec_id, batch_inputs, outputs[vec_id], &batch_labels[vec_id]);

    loss_fn.lossGradients(outputs[vec_id], batch_labels[vec_id],
                          batch_inputs.getBatchSize());

    backpropagate(vec_id, batch_inputs, outputs[vec_id]);

    metrics.processSample(outputs[vec_id], batch_labels[vec_id]);
  }

  updateParameters(learning_rate, ++_batch_iter);
}

template <typename BATCH_T>
inline void Model<BATCH_T>::updateSampling(uint32_t rehash_batch,
                                           uint32_t rebuild_batch) {
  if (checkBatchInterval(rebuild_batch)) {
    reBuildHashFunctions();
    buildHashTables();
  } else if (checkBatchInterval(rehash_batch)) {
    buildHashTables();
  }
}

template <typename BATCH_T>
InferenceMetricData Model<BATCH_T>::predict(
    const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& test_data,
    const dataset::BoltDatasetPtr& labels, uint32_t* output_active_neurons,
    float* output_activations, bool use_sparse_inference,
    const std::vector<std::string>& metric_names, bool verbose,
    uint32_t batch_limit) {
  assert(output_activations != nullptr || output_active_neurons == nullptr);
  bool compute_metrics = labels != nullptr;

  uint32_t batch_size = test_data->at(0).getBatchSize();

  uint64_t num_test_batches =
      std::min<uint64_t>(test_data->numBatches(), batch_limit);

  uint64_t inference_output_dim = getInferenceOutputDim(use_sparse_inference);

  MetricAggregator metrics(metric_names, verbose);

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  initializeNetworkState(batch_size, /* use_sparsity= */ use_sparse_inference);
  BoltBatch outputs =
      getOutputs(batch_size, /* use_sparsity= */ use_sparse_inference);

  ProgressBar bar(num_test_batches, verbose);

  auto test_start = std::chrono::high_resolution_clock::now();
  for (uint32_t batch = 0; batch < num_test_batches; batch++) {
    const BATCH_T& inputs = test_data->at(batch);

    const BoltBatch* batch_labels =
        compute_metrics ? &(*labels)[batch] : nullptr;

    uint32_t* batch_active_neurons =
        output_active_neurons == nullptr
            ? nullptr
            : output_active_neurons + batch * batch_size * inference_output_dim;

    float* batch_activations =
        output_activations == nullptr
            ? nullptr
            : output_activations + batch * batch_size * inference_output_dim;

    processTestBatch(inputs, outputs, batch_labels, batch_active_neurons,
                     batch_activations, inference_output_dim, metrics,
                     compute_metrics);

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
InferenceMetricData Model<BATCH_T>::predictOnStream(
    const std::shared_ptr<dataset::StreamingDataset<BATCH_T, BoltBatch>>&
        test_data,
    bool use_sparse_inference, const std::vector<std::string>& metric_names,
    std::optional<std::function<void(const bolt::BoltBatch&, uint32_t)>>
        batch_callback,
    bool verbose) {
  MetricAggregator metrics(metric_names, verbose);

  uint32_t batch_size = test_data->getMaxBatchSize();

  uint64_t inference_output_dim = getInferenceOutputDim(use_sparse_inference);

  initializeNetworkState(batch_size, /* use_sparsity= */ use_sparse_inference);
  BoltBatch outputs =
      getOutputs(batch_size, /* use_sparsity= */ use_sparse_inference);

  if (verbose) {
    std::cout << std::endl
              << "Processing test streaming dataset..." << std::endl;
  }

  auto test_start = std::chrono::high_resolution_clock::now();

  while (auto batch = test_data->nextBatchTuple()) {
    processTestBatch(
        std::get<0>(batch.value()), outputs, &std::get<1>(batch.value()),
        /* output_active_neurons=*/nullptr,
        /* output_activations=*/nullptr, inference_output_dim, metrics,
        /* compute_metrics= */ true);

    if (batch_callback) {
      batch_callback.value()(outputs,
                             std::get<0>(batch.value()).getBatchSize());
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
inline void Model<BATCH_T>::processTestBatch(
    const BATCH_T& batch_inputs, BoltBatch& outputs,
    const BoltBatch* batch_labels, uint32_t* output_active_neurons,
    float* output_activations, uint64_t inference_output_dim,
    MetricAggregator& metrics, bool compute_metrics) {
#pragma omp parallel for default(none)                                 \
    shared(batch_inputs, batch_labels, outputs, output_active_neurons, \
           output_activations, inference_output_dim, metrics, compute_metrics)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    // We set labels to nullptr so that they are not used in sampling during
    // inference.
    forward(vec_id, batch_inputs, outputs[vec_id], /*labels=*/nullptr);

    if (compute_metrics) {
      metrics.processSample(outputs[vec_id], (*batch_labels)[vec_id]);
    }

    if (output_activations != nullptr) {
      assert(outputs[vec_id].len == inference_output_dim);
      const float* start = outputs[vec_id].activations;
      uint32_t offset = vec_id * inference_output_dim;
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