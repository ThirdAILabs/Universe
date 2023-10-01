#include "Mach.h"
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/metrics/Metric.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/MachLabel.h>
#include <tuple>

namespace thirdai::automl::udt::utils {

std::tuple<feat::MachLabelPtr, feat::OutputColumnsList> Mach::machTransform(
    const std::string& doc_id_column) {
  auto transform = feat::MachLabel::make(doc_id_column, MACH_LABELS);
  feat::OutputColumnsList output_columns{feat::OutputColumns(MACH_LABELS),
                                         feat::OutputColumns(doc_id_column)};
  return std::make_tuple(std::move(transform), std::move(output_columns));
}

void Mach::introduceEntities(const feat::ColumnMap& table,
                             const feat::OutputColumnsList& input_columns,
                             const std::string& doc_id_column,
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
  auto inputs = feat::toTensors(table, input_columns);
  auto scores = _model->forward(inputs).at(0);

  ctrl_c_check();

  auto doc_ids = table.getArrayColumn<uint32_t>(doc_id_column);
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

bolt::metrics::History Mach::train(feat::ColumnMapIteratorPtr train_iter,
                                   feat::ColumnMapIteratorPtr valid_iter,
                                   const feat::OutputColumnsList& input_columns,
                                   const std::string& doc_id_column,
                                   float learning_rate, uint32_t epochs,
                                   const InputMetrics& train_metrics,
                                   const InputMetrics& val_metrics,
                                   const std::vector<CallbackPtr>& callbacks,
                                   TrainOptions options,
                                   const bolt::DistributedCommPtr& comm) {
  auto [mach_transform, label_columns] = machTransform(doc_id_column);

  auto train_loader = feat::Loader::make(
      std::move(train_iter), mach_transform, _state, input_columns,
      label_columns,
      /* batch_size= */ options.batch_size.value_or(defaults::BATCH_SIZE),
      /* shuffle= */ true,
      /* verbose= */ options.verbose,
      /* shuffle_buffer_size= */ options.shuffle_config.min_buffer_size,
      /* shuffle_seed= */ options.shuffle_config.seed);

  feat::LoaderPtr valid_loader;
  if (valid_iter) {
    valid_loader = feat::Loader::make(std::move(valid_iter), mach_transform,
                                      _state, input_columns, label_columns,
                                      /* batch_size= */ defaults::BATCH_SIZE,
                                      /* shuffle= */ false,
                                      /* verbose= */ options.verbose);
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
      /* validation_data_loader= */ valid_loader,
      /* validation_metrics= */ val_metrics,
      /* steps_per_validation= */ options.steps_per_validation,
      /* use_sparsity_in_validation= */ options.sparse_validation,
      /* callbacks= */ callbacks, /* autotune_rehash_rebuild= */ true,
      /* verbose= */ options.verbose,
      /* logging_interval= */ options.logging_interval,
      /*comm= */ comm);
}

void Mach::trainBatch(feat::ColumnMap table,
                      const feat::OutputColumnsList& input_columns,
                      const std::string& doc_id_column, float learning_rate) {
  auto [mach_transform, label_columns] = machTransform(doc_id_column);
  table = mach_transform->apply(std::move(table), *_state);
  auto inputs = feat::toTensors(table, input_columns);
  auto labels = feat::toTensors(table, label_columns);
  _model->trainOnBatch(inputs, labels);
  _model->updateParameters(learning_rate);
}

bolt::metrics::History Mach::evaluate(
    feat::ColumnMapIteratorPtr eval_iter,
    const feat::OutputColumnsList& input_columns,
    const std::string& doc_id_column, const InputMetrics& metrics,
    bool sparse_inference, bool verbose) {
  auto [mach_transform, label_columns] = machTransform(doc_id_column);

  auto valid_loader = feat::Loader::make(std::move(eval_iter), mach_transform,
                                         _state, input_columns, label_columns,
                                         /* batch_size= */ defaults::BATCH_SIZE,
                                         /* shuffle= */ false,
                                         /* verbose= */ verbose);

  bolt::Trainer trainer(_model, std::nullopt, bolt::python::CtrlCCheck{});
  auto history = trainer.validate_with_data_loader(valid_loader, metrics,
                                                   sparse_inference, verbose);

  return history;
}

std::vector<std::vector<std::pair<uint32_t, double>>> Mach::predict(
    const feat::ColumnMap& table, const feat::OutputColumnsList& input_columns,
    bool sparse_inference, uint32_t top_k, uint32_t num_scanned_buckets) {
  auto inputs = feat::toTensors(table, input_columns);
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

auto teach(bolt::ModelPtr& model, feat::ColumnMap rlhf_table,
           feat::ColumnMap balancing_table,
           const feat::OutputColumnsList& input_columns,
           const feat::OutputColumnsList& label_columns, float learning_rate,
           uint32_t repeats, uint32_t epochs, size_t batch_size,
           const InputMetrics& metrics = {}, bool verbose = false,
           std::optional<uint32_t> logging_interval = std::nullopt) {
  rlhf_table = repeatRows(std::move(rlhf_table), repeats);

  auto all_columns = feat::allColumns(input_columns, label_columns);
  rlhf_table = feat::keepColumns(std::move(rlhf_table), all_columns);
  balancing_table = feat::keepColumns(std::move(balancing_table), all_columns);
  auto train_table = rlhf_table.concat(balancing_table);

  bolt::Trainer trainer(model);
  return trainer.train(
      /* train_data= */ thirdai::data::toLabeledDataset(
          /* table= */ train_table, /* input_columns= */ input_columns,
          /* label_columns= */ label_columns, /* batch_size= */ batch_size),
      /* learning_rate= */ learning_rate,
      /* epochs= */ epochs,
      /* train_metrics= */ metrics,
      /* validation_data= */ {},
      /* validation_metrics= */ {},
      /* steps_per_validation= */ {},
      /* use_sparsity_in_validation= */ false,
      /* callbacks= */ {},
      /* autotune_rehash_rebuild= */ true,
      /* verbose= */ verbose,
      /* logging_interval= */ logging_interval);
}

void Mach::upvote(feat::ColumnMap upvotes, feat::ColumnMap balancers,
                  const feat::OutputColumnsList& input_columns,
                  const std::string& doc_id_column, float learning_rate,
                  uint32_t repeats, uint32_t epochs, size_t batch_size) {
  auto [mach_transform, label_columns] = machTransform(doc_id_column);
  upvotes = mach_transform->apply(std::move(upvotes), *_state);
  balancers = mach_transform->apply(std::move(balancers), *_state);

  teach(_model, std::move(upvotes), std::move(balancers), input_columns,
        label_columns, learning_rate, repeats, epochs, batch_size);
}

auto predictBuckets(bolt::Model& model, const dataset::mach::MachIndex& index,
                    const feat::ColumnMap& table,
                    const feat::OutputColumnsList& input_columns,
                    std::optional<uint32_t> num_buckets,
                    bool sparse_inference = false) {
  auto inputs = feat::toTensors(table, input_columns);
  auto outputs = model.forward(inputs, sparse_inference).at(0);

  std::vector<std::vector<uint32_t>> all_hashes(outputs->batchSize());
#pragma omp parallel for default(none) \
    shared(outputs, all_hashes, index, num_buckets)
  for (uint32_t i = 0; i < outputs->batchSize(); i++) {
    const BoltVector& output = outputs->getVector(i);

    TopKActivationsQueue heap;
    heap = index.topKNonEmptyBuckets(output,
                                     num_buckets.value_or(index.numHashes()));

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

bolt::metrics::History Mach::associate(
    feat::ColumnMap from_table, const feat::ColumnMap& to_table,
    feat::ColumnMap balancing_table,
    const feat::OutputColumnsList& input_columns,
    const std::string& doc_id_column, float learning_rate, uint32_t repeats,
    uint32_t num_buckets, uint32_t epochs, size_t batch_size,
    const InputMetrics& metrics, bool verbose,
    std::optional<uint32_t> logging_interval) {
  auto rlhf_table = std::move(from_table);

  auto dummy_doc_ids = thirdai::data::ValueColumn<uint32_t>::make(
      std::vector<uint32_t>(from_table.numRows(), 0),
      std::numeric_limits<uint32_t>::max());
  rlhf_table.setColumn(doc_id_column, dummy_doc_ids);

  auto mach_labels = thirdai::data::ArrayColumn<uint32_t>::make(
      predictBuckets(*_model, *index(), to_table, input_columns, num_buckets),
      index()->numBuckets());
  rlhf_table.setColumn(MACH_LABELS, mach_labels);

  auto [mach_transform, label_columns] = machTransform(doc_id_column);
  balancing_table = mach_transform->apply(std::move(balancing_table), *_state);

  return teach(_model, std::move(rlhf_table), std::move(balancing_table),
               input_columns, label_columns, learning_rate, repeats, epochs,
               batch_size, metrics, verbose, logging_interval);
}

std::vector<uint32_t> Mach::outputCorrectness(
    const feat::ColumnMap& table, const feat::OutputColumnsList& input_columns,
    const std::vector<uint32_t>& labels, std::optional<uint32_t> num_hashes,
    bool sparse_inference) {
  auto top_buckets = predictBuckets(*_model, *index(), table, input_columns,
                                    num_hashes, sparse_inference);

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

bolt::TensorPtr Mach::embedding(const feat::ColumnMap& table,
                                const feat::OutputColumnsList& input_columns) {
  // TODO(Nicholas): Sparsity could speed this up, and wouldn't affect the
  // embeddings if the sparsity is in the output layer and the embeddings are
  // from the hidden layer.
  auto inputs = feat::toTensors(table, input_columns);
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
