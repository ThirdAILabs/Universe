#include "Mach.h"
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/MachLabel.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <cstddef>
#include <tuple>
#include <vector>

namespace thirdai::automl::udt::utils {

void Mach::introduceEntities(const feat::ColumnMap& table,
                             std::optional<uint32_t> num_buckets_to_sample_opt,
                             uint32_t num_random_hashes) {
  uint32_t num_buckets_to_sample =
      num_buckets_to_sample_opt.value_or(index()->numHashes());

  std::unordered_map<uint32_t, std::vector<TopKActivationsQueue>> top_k_per_doc;

  bolt::python::CtrlCCheck ctrl_c_check;

  // Note: using sparse inference here could cause issues because the
  // mach index sampler will only return nonempty buckets, which could
  // cause new docs to only be mapped to buckets already containing
  // entities.
  auto inputs = feat::toTensors(table, _bolt_input_columns);
  auto scores = _model->forward(inputs).at(0);

  ctrl_c_check();

  auto doc_ids = table.getArrayColumn<uint32_t>(labelColumn());
  for (uint32_t row = 0; row < scores->batchSize(); row++) {
    uint32_t label = doc_ids->row(row)[0];
    top_k_per_doc[label].push_back(
        scores->getVector(row).findKLargestActivations(num_buckets_to_sample));
    ctrl_c_check();
  }

  for (auto& [doc, top_ks] : top_k_per_doc) {
    auto hashes = topHashesForDoc(*index(), std::move(top_ks),
                                  num_buckets_to_sample, num_random_hashes);
    index()->insert(doc, hashes);

    ctrl_c_check();
  }

  updateSamplingStrategy();
}

bolt::metrics::History Mach::train(
    feat::ColumnMapIteratorPtr train_iter, feat::ColumnMapIteratorPtr val_iter,
    float learning_rate, uint32_t epochs, const InputMetrics& train_metrics,
    const InputMetrics& val_metrics, const std::vector<CallbackPtr>& callbacks,
    TrainOptions options, const bolt::DistributedCommPtr& comm) {
  auto train_loader = getDataLoader(
      std::move(train_iter), /* store_balancers= */ true, options.batchSize(),
      /* shuffle= */ true, options.verbose, options.shuffle_config);

  feat::LoaderPtr val_loader;
  if (val_iter) {
    val_loader = getDataLoader(
        std::move(val_iter), /* store_balancers= */ false, defaults::BATCH_SIZE,
        /* shuffle= */ false, options.verbose);
  }

  std::optional<uint32_t> freeze_hash_tables_epoch = std::nullopt;
  if (_freeze_hash_tables) {
    freeze_hash_tables_epoch = 1;
  }

  bolt::Trainer trainer(_model, freeze_hash_tables_epoch,
                        bolt::python::CtrlCCheck{});

  return trainer.train_with_data_loader(
      /* train_data_loader= */ train_loader,
      /* learning_rate= */ learning_rate, /* epochs= */ epochs,
      /* max_in_memory_batches= */ options.max_in_memory_batches,
      /* train_metrics= */ train_metrics,
      /* validation_data_loader= */ val_loader,
      /* validation_metrics= */ val_metrics,
      /* steps_per_validation= */ options.steps_per_validation,
      /* use_sparsity_in_validation= */ options.sparse_validation,
      /* callbacks= */ callbacks, /* autotune_rehash_rebuild= */ true,
      /* verbose= */ options.verbose,
      /* logging_interval= */ options.logging_interval,
      /*comm= */ comm);
}

void Mach::train(feat::ColumnMap table, float learning_rate) {
  table = _label_to_buckets->apply(std::move(table), *_state);
  _store_rlhf_samples->apply(table, *_state);
  auto inputs = feat::toTensors(table, _bolt_input_columns);
  auto labels = feat::toTensors(table, _bolt_label_columns);
  _model->trainOnBatch(inputs, labels);
  _model->updateParameters(learning_rate);
}

bolt::metrics::History Mach::evaluate(feat::ColumnMapIteratorPtr eval_iter,
                                      const InputMetrics& metrics,
                                      bool sparse_inference, bool verbose) {
  auto eval_loader = getDataLoader(
      std::move(eval_iter), /* store_balancers= */ false,
      /* batch_size= */ defaults::BATCH_SIZE, /* shuffle= */ false, verbose);

  bolt::Trainer trainer(_model, std::nullopt, bolt::python::CtrlCCheck{});
  auto history = trainer.validate_with_data_loader(eval_loader, metrics,
                                                   sparse_inference, verbose);

  return history;
}

std::vector<std::vector<std::pair<uint32_t, double>>> Mach::predict(
    const feat::ColumnMap& table, bool sparse_inference, uint32_t top_k,
    uint32_t num_scanned_buckets) {
  auto inputs = feat::toTensors(table, _bolt_input_columns);
  auto outputs = _model->forward(inputs, sparse_inference).at(0);

  uint32_t num_classes = index()->numEntities();
  if (top_k && top_k > num_classes) {
    throw std::invalid_argument(
        "Cannot return more results than the model is trained to "
        "predict. "
        "Model currently can predict one of " +
        std::to_string(num_classes) + " classes.");
  }

  uint32_t batch_size = outputs->batchSize();

  std::vector<std::vector<std::pair<uint32_t, double>>> predicted_entities(
      batch_size);
#pragma omp parallel for default(none)                     \
    shared(outputs, predicted_entities, top_k, batch_size, \
           num_scanned_buckets) if (batch_size > 1)
  for (uint32_t i = 0; i < batch_size; i++) {
    const BoltVector& vector = outputs->getVector(i);
    auto predictions = index()->decode(
        /* output = */ vector,
        /* top_k = */ top_k,
        /* num_buckets_to_eval = */ num_scanned_buckets);
    predicted_entities[i] = predictions;
  }

  return predicted_entities;
}

auto repeatRows(feat::ColumnMap&& table, uint32_t repetitions) {
  std::vector<size_t> permutation(table.numRows() * repetitions);
  for (uint32_t rep = 0; rep < repetitions; rep++) {
    auto begin = permutation.begin() + rep * table.numRows();
    std::iota(begin, begin + table.numRows(), 0);
  }
  return table.permute(permutation);
}

void Mach::teach(feat::ColumnMap feedback, float learning_rate,
                 uint32_t feedback_repetitions, uint32_t num_balancers,
                 uint32_t epochs, size_t batch_size) {
  feedback = repeatRows(std::move(feedback), feedback_repetitions);
  feedback = feat::keepColumns(std::move(feedback), _all_bolt_columns);

  auto balancers = _state->rlhfSampler()->balancingSamples(num_balancers);
  balancers = feat::keepColumns(std::move(balancers), _all_bolt_columns);

  auto train_table = feedback.concat(balancers);

  auto train_data = thirdai::data::toLabeledDataset(
      train_table, _bolt_input_columns, _bolt_label_columns, batch_size);

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    for (uint32_t batch = 0; batch < train_data.first.size(); batch++) {
      _model->trainOnBatch(train_data.first.at(batch),
                           train_data.second.at(batch));
      _model->updateParameters(learning_rate);
    }
  }
}

void Mach::associate(feat::ColumnMap from_table,
                     const feat::ColumnMap& to_table, float learning_rate,
                     uint32_t repeats, uint32_t num_balancers,
                     uint32_t num_buckets, uint32_t epochs, size_t batch_size) {
  auto mach_labels = thirdai::data::ArrayColumn<uint32_t>::make(
      predictBuckets(to_table, /* sparse_inference= */ false, num_buckets),
      index()->numBuckets());
  from_table.setColumn(MACH_LABELS, mach_labels);
  from_table = addDummyLabels(std::move(from_table));
  return teach(std::move(from_table), learning_rate, repeats, num_balancers,
               epochs, batch_size);
}

bolt::metrics::History Mach::associateTrain(
    feat::ColumnMap from_table, const feat::ColumnMap& to_table,
    data::ColumnMap train_data, float learning_rate, uint32_t repeats,
    uint32_t num_buckets, uint32_t epochs, size_t batch_size,
    const InputMetrics& metrics, TrainOptions options) {
  auto mach_labels = thirdai::data::ArrayColumn<uint32_t>::make(
      predictBuckets(to_table, /* sparse_inference= */ false, num_buckets),
      index()->numBuckets());
  from_table.setColumn(MACH_LABELS, mach_labels);
  from_table = addDummyLabels(std::move(from_table));
  from_table = repeatRows(std::move(from_table), repeats);
  from_table = data::keepColumns(std::move(from_table), _all_bolt_columns);

  train_data = _label_to_buckets->apply(std::move(train_data), *_state);
  _store_rlhf_samples->apply(train_data, *_state);
  train_data = data::keepColumns(std::move(train_data), _all_bolt_columns);
  train_data = train_data.concat(from_table);

  bolt::Trainer trainer(_model);
  return trainer.train(
      /* train_data= */ thirdai::data::toLabeledDataset(
          train_data, _bolt_input_columns, _bolt_label_columns, batch_size),
      /* learning_rate= */ learning_rate,
      /* epochs= */ epochs,
      /* train_metrics= */ metrics,
      /* validation_data= */ {},
      /* validation_metrics= */ {},
      /* steps_per_validation= */ {},
      /* use_sparsity_in_validation= */ false,
      /* callbacks= */ {},
      /* autotune_rehash_rebuild= */ true,
      /* verbose= */ options.verbose,
      /* logging_interval= */ options.logging_interval);
}

void Mach::upvote(feat::ColumnMap upvotes, float learning_rate,
                  uint32_t repeats, uint32_t num_balancers, uint32_t epochs,
                  size_t batch_size) {
  upvotes = _label_to_buckets->apply(std::move(upvotes), *_state);
  teach(std::move(upvotes), learning_rate, repeats, num_balancers, epochs,
        batch_size);
}

void Mach::trainBuckets(feat::ColumnMap table, float learning_rate) {
  table = addDummyLabels(std::move(table));
  auto inputs = feat::toTensors(table, _bolt_input_columns);
  auto labels = feat::toTensors(table, _bolt_label_columns);
  _model->trainOnBatch(inputs, labels);
  _model->updateParameters(learning_rate);
}

std::vector<std::vector<uint32_t>> Mach::predictBuckets(
    const feat::ColumnMap& table, bool sparse_inference,
    std::optional<uint32_t> top_k) {
  auto inputs = feat::toTensors(table, _bolt_input_columns);
  auto outputs = _model->forward(inputs, sparse_inference).at(0);

  std::vector<std::vector<uint32_t>> all_hashes(outputs->batchSize());
#pragma omp parallel for default(none) shared(outputs, all_hashes, top_k)
  for (uint32_t i = 0; i < outputs->batchSize(); i++) {
    const BoltVector& output = outputs->getVector(i);

    TopKActivationsQueue heap;
    heap = _state->machIndex()->topKNonEmptyBuckets(
        output, top_k.value_or(_state->machIndex()->numHashes()));

    std::vector<uint32_t> hashes;
    while (!heap.empty()) {
      auto [_, active_neuron] = heap.top();
      hashes.push_back(active_neuron);
      heap.pop();
    }

    std::reverse(hashes.begin(), hashes.end());

    all_hashes[i] = hashes;
  }

  return all_hashes;
}

std::vector<uint32_t> Mach::outputCorrectness(
    const feat::ColumnMap& table, const std::vector<uint32_t>& labels,
    std::optional<uint32_t> num_hashes, bool sparse_inference) {
  auto top_buckets = predictBuckets(table, sparse_inference, num_hashes);

  std::vector<uint32_t> matching_buckets(labels.size());
  std::exception_ptr hashes_err;

#pragma omp parallel for default(none) \
    shared(labels, top_buckets, matching_buckets, hashes_err)
  for (uint32_t i = 0; i < labels.size(); i++) {
    try {
      std::vector<uint32_t> hashes = index()->getHashes(labels[i]);
      uint32_t count = 0;
      for (auto hash : hashes) {
        if (std::count(top_buckets[i].begin(), top_buckets[i].end(), hash) >
            0) {
          count++;
        }
      }
      matching_buckets[i] = count;
    } catch (const std::exception& e) {
#pragma omp critical
      hashes_err = std::current_exception();
    }
  }

  if (hashes_err) {
    std::rethrow_exception(hashes_err);
  }

  return matching_buckets;
}

bolt::TensorPtr Mach::embedding(const feat::ColumnMap& table) {
  // TODO(Nicholas): Sparsity could speed this up, and wouldn't affect the
  // embeddings if the sparsity is in the output layer and the embeddings are
  // from the hidden layer.
  auto inputs = feat::toTensors(table, _bolt_input_columns);
  _model->forward(inputs, /* use_sparsity= */ false);
  return _emb->tensor();
}

std::vector<float> Mach::entityEmbedding(uint32_t entity) const {
  std::vector<uint32_t> buckets = index()->getHashes(entity);

  auto outputs = _model->outputs();

  if (outputs.size() != 1) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }
  auto fc = bolt::FullyConnected::cast(outputs.at(0)->op());
  if (!fc) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }

  auto fc_layer = fc->kernel();

  std::vector<float> averaged_embedding(fc_layer->getInputDim());
  for (uint32_t neuron_id : buckets) {
    auto weights = fc_layer->getWeightsByNeuron(neuron_id);
    if (weights.size() != averaged_embedding.size()) {
      throw std::invalid_argument("Output dim mismatch.");
    }
    for (uint32_t i = 0; i < weights.size(); i++) {
      averaged_embedding[i] += weights[i];
    }
  }

  // TODO(david) try averaging and summing
  for (float& weight : averaged_embedding) {
    weight /= averaged_embedding.size();
  }

  return averaged_embedding;
}

void Mach::updateSamplingStrategy() {
  auto output_layer =
      bolt::FullyConnected::cast(_model->opExecutionOrder().back());

  const auto& neuron_index = output_layer->kernel()->neuronIndex();

  float index_sparsity = index()->sparsity();
  if (index_sparsity > 0 && index_sparsity <= _mach_sampling_threshold) {
    // TODO(Nicholas) add option to specify new neuron index in set sparsity.
    output_layer->setSparsity(index_sparsity, false, false);
    auto new_index = bolt::MachNeuronIndex::make(index());
    output_layer->kernel()->setNeuronIndex(new_index);

  } else {
    if (std::dynamic_pointer_cast<bolt::MachNeuronIndex>(neuron_index)) {
      float sparsity = utils::autotuneSparsity(index()->numBuckets());

      auto sampling_config = bolt::DWTASamplingConfig::autotune(
          index()->numBuckets(), sparsity,
          /* experimental_autotune= */ false);

      output_layer->setSparsity(sparsity, false, false);

      if (sampling_config) {
        auto new_index = sampling_config->getNeuronIndex(
            output_layer->dim(), output_layer->inputDim());
        output_layer->kernel()->setNeuronIndex(new_index);
      }
    }
  }
}

}  // namespace thirdai::automl::udt::utils
