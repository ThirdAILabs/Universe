#include "Model.h"
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
  initializeNetworkState(batch_size, false);
  BoltBatch outputs = getOutputs(batch_size, false);

  std::vector<double> time_per_epoch;

  MetricAggregator metrics(metric_names, verbose);

  // if any layer is shallow, call enable training to set the optimizer state
  // before training
  if (anyLayerShallow()) {
    throw std::logic_error(
        "Call reinitialize_optimizer_for_training before training to "
        "initialize optimizer state");
  }

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
MetricData Model<BATCH_T>::trainOnStream(
    std::shared_ptr<dataset::StreamingDataset<BATCH_T>>& train_data,
    const LossFunction& loss_fn, float learning_rate, uint32_t rehash_batch,
    uint32_t rebuild_batch, const std::vector<std::string>& metric_names,
    uint32_t metric_log_batch_interval, bool verbose) {
  MetricAggregator metrics(metric_names, verbose);

  uint32_t batch_size = train_data->getMaxBatchSize();
  initializeNetworkState(batch_size, /* force_dense=*/false);
  BoltBatch outputs = getOutputs(batch_size, /* force_dense=*/false);

  if (verbose) {
    std::cout << std::endl
              << "Processing training streaming dataset..." << std::endl;
  }

  auto train_start = std::chrono::high_resolution_clock::now();

  uint32_t batch_count = 0;
  while (auto batch = train_data->nextBatch()) {
    processTrainingBatch(
        /* batch_inputs=*/batch->first, /* outputs= */ outputs,
        /* batch_labels= */ batch->second, /* loss_fn= */ loss_fn,
        /* learning_rate= */ learning_rate, /* rehash_batch */ rehash_batch,
        /* rebuild_batch= */ rebuild_batch, /* metrics= */ metrics);

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

    loss_fn.lossGradients(outputs[vec_id], batch_labels[vec_id],
                          batch_inputs.getBatchSize());

    backpropagate(vec_id, batch_inputs, outputs[vec_id]);

    metrics.processSample(outputs[vec_id], batch_labels[vec_id]);
  }

  updateParameters(learning_rate, ++_batch_iter);

  if (_batch_iter % rebuild_batch == (rebuild_batch - 1)) {
    reBuildHashFunctions();
    buildHashTables();
  } else if (_batch_iter % rehash_batch == (rehash_batch - 1)) {
    buildHashTables();
  }
}
inline void getMaxandSecondMax(const float* activations, uint32_t dim,
                               uint32_t& max_index,
                               uint32_t& second_max_index) {
  float first = std::numeric_limits<float>::min(),
        second = std::numeric_limits<float>::min();
  uint32_t max = 0, second_max = 0;
  if (dim < 2) {
    throw std::invalid_argument("cant do this process");
  }
  for (uint32_t i = 0; i < dim; i++) {
    if (activations[i] > first) {
      second = first;
      second_max = max;
      first = activations[i];
      max = i;
    } else if (activations[i] > second && activations[i] != first) {
      second = activations[i];
      second_max = i;
    }
  }
  max_index = max, second_max_index = second_max;
}
template <typename BATCH_T>
inline std::vector<float> Model<BATCH_T>::getInputGradientsFromModel(
    std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& batch_input,
    const LossFunction& loss_fn) {
  uint64_t num_batches = batch_input->numBatches();
  std::vector<float> total_grad;
  for (uint64_t r = 0; r < num_batches; r++) {
    // Because of how the datasets are read we know that all batches will not
    // have a batch size larger than this so we can just set the batch size
    // here.
    initializeNetworkState(batch_input->at(r).getBatchSize(), false);
    BoltBatch output = getOutputs(batch_input->at(r).getBatchSize(), false);
    for (uint32_t vec_id = 0; vec_id < batch_input->at(r).getBatchSize();
         vec_id++) {
      // Initializing the input gradients, because they were not initialized
      // before.
      batch_input->at(r)[vec_id].gradients =
          new float[batch_input->at(r)[vec_id].len];
      forward(vec_id, batch_input->at(r), output[vec_id], nullptr);
      uint32_t max_index, second_max_index;
      getMaxandSecondMax(output[vec_id].activations, getOutputDim(), max_index,
                         second_max_index);
      // backpropagating twice with different output labels one with highest activation and another
      // second highest activation and getting the difference of input gradients.
      BoltVector batch_label_first = BoltVector::makeSparseVector(
          std::vector<uint32_t>{max_index}, std::vector<float>{1.0});
      BoltVector batch_label_second = BoltVector::makeSparseVector(
          std::vector<uint32_t>{second_max_index}, std::vector<float>{1.0});
      loss_fn.lossGradients(output[vec_id], batch_label_first,
                            batch_input->at(r).getBatchSize());
      std::vector<float> input_gradients_first =
          backpropagateInput(vec_id, batch_input->at(r), output[vec_id]);
      loss_fn.lossGradients(output[vec_id], batch_label_second,
                            batch_input->at(r).getBatchSize());
      std::vector<float> input_gradients_second =
          backpropagateInput(vec_id, batch_input->at(r), output[vec_id]);
      for (uint32_t i = 0; i < batch_input->at(r)[vec_id].len; i++) {
        input_gradients_second[i] = input_gradients_second[i] - input_gradients_first[i];
        total_grad.push_back(input_gradients_second[i]);
      }
    }
  }
  return total_grad;
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
InferenceMetricData Model<BATCH_T>::predictOnStream(
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
    processTestBatch(batch->first, outputs, &batch->second,
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
inline void Model<BATCH_T>::processTestBatch(const BATCH_T& batch_inputs,
                                             BoltBatch& outputs,
                                             const BoltBatch* batch_labels,
                                             uint32_t* output_active_neurons,
                                             float* output_activations,
                                             MetricAggregator& metrics,
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