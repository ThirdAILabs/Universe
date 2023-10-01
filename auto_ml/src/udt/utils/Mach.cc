#include "Mach.h"
#include <bolt/src/nn/model/Model.h>

namespace thirdai::automl::udt::utils {

const std::string& docIdColumn(const OutputColumnsList& columns) {
  if (columns.size() > 1) {
    throw std::runtime_error("Mach only supports one label column.");
  }
  if (columns.front().values()) {
    throw std::runtime_error("Mach does not support weighted labels.");
  }

  return columns.front().indices();
}

TransformedTable tableWithMachTransform(
    TransformedTable table, dataset::mach::MachIndexPtr mach_index) {
  if (!table.labels) {
    return table;
  }

  auto state = State::make(std::move(mach_index));
  auto doc_id_column = docIdColumn(*table.labels);
  table.table =
      MachLabel(doc_id_column, MACH_LABELS).apply(table.table, *state);
  // Insert MACH_LABELS into first position because this is what the bolt model
  // expects.
  table.labels->insert(table.labels->begin(), OutputColumns(MACH_LABELS));
  return table;
}

TransformedIterator iterWithMachTransform(
    TransformedIterator iter, dataset::mach::MachIndexPtr mach_index) {
  if (!iter.labels) {
    return iter;
  }

  auto state = State::make(std::move(mach_index));
  auto doc_id_column = docIdColumn(*iter.labels);
  auto mach_transform = MachLabel::make(doc_id_column, MACH_LABELS);
  iter.iter =
      TransformIterator::make(std::move(iter.iter), mach_transform, state);
  // Insert MACH_LABELS into first position because this is what the bolt model
  // expects.
  iter.labels->insert(iter.labels->begin(), OutputColumns(MACH_LABELS));
  return iter;
}

void Mach::introduceEntities(const TransformedTable& table,
                             std::optional<uint32_t> num_buckets_to_sample_opt,
                             uint32_t num_random_hashes) {
  TransformedTensors tensors(table);
  const auto& label_tensor = tensors.labels->front();

  uint32_t num_buckets_to_sample =
      num_buckets_to_sample_opt.value_or(_mach_index->numHashes());

  std::unordered_map<uint32_t, std::vector<TopKActivationsQueue>> top_k_per_doc;

  bolt::python::CtrlCCheck ctrl_c_check;

  // Note: using sparse inference here could cause issues because the
  // mach index sampler will only return nonempty buckets, which could
  // cause new docs to only be mapped to buckets already containing
  // entities.
  auto scores = _model->forward(tensors.inputs).at(0);

  ctrl_c_check();

  for (uint32_t row = 0; row < scores->batchSize(); row++) {
    uint32_t label = label_tensor->getVector(row).active_neurons[0];
    top_k_per_doc[label].push_back(
        scores->getVector(row).findKLargestActivations(num_buckets_to_sample));
    ctrl_c_check();
  }

  for (auto& [doc, top_ks] : top_k_per_doc) {
    auto hashes = topHashesForDoc(*_mach_index, std::move(top_ks),
                                  num_buckets_to_sample, num_random_hashes);
    _mach_index->insert(doc, hashes);

    ctrl_c_check();
  }

  updateSamplingStrategy();
}

auto Mach::train(TransformedIterator train_data,
                 std::optional<TransformedIterator> valid_data,
                 float learning_rate, uint32_t epochs,
                 const InputMetrics& train_metrics,
                 const InputMetrics& val_metrics,
                 const std::vector<CallbackPtr>& callbacks,
                 TrainOptions options, const bolt::DistributedCommPtr& comm) {
  auto train = iterWithMachTransform(std::move(train_data), _mach_index);
  auto train_loader = Loader::make(
      /* data_iterator= */ train.iter,
      /* input_columns= */ train.inputs,
      /* label_columns= */ train.labels.value(),
      /* batch_size= */ options.batch_size.value_or(defaults::BATCH_SIZE),
      /* shuffle= */ true,
      /* verbose= */ options.verbose,
      /* shuffle_buffer_size= */ options.shuffle_config.min_buffer_size,
      /* shuffle_seed= */ options.shuffle_config.seed);

  LoaderPtr valid_loader;
  if (valid_data) {
    auto valid = iterWithMachTransform(std::move(*valid_data), _mach_index);
    valid_loader = Loader::make(
        /* data_iterator= */ valid.iter,
        /* input_columns= */ valid.inputs,
        /* label_columns= */ valid.labels.value(),
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

  auto history = trainer.train_with_data_loader(
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

  return history;
}

void Mach::trainBatch(TransformedTable batch, float learning_rate) {
  batch = tableWithMachTransform(std::move(batch), _mach_index);
  TransformedTensors tensors(batch);
  _model->trainOnBatch(tensors.inputs, tensors.labels.value());
  _model->updateParameters(learning_rate);
}

auto Mach::evaluate(TransformedIterator eval_data, const InputMetrics& metrics,
                    bool sparse_inference, bool verbose) {
  auto eval = iterWithMachTransform(std::move(eval_data), _mach_index);

  auto valid_loader = Loader::make(
      /* data_iterator= */ eval.iter,
      /* input_columns= */ eval.inputs,
      /* label_columns= */ eval.labels.value(),
      /* batch_size= */ defaults::BATCH_SIZE,
      /* shuffle= */ false,
      /* verbose= */ verbose);

  bolt::Trainer trainer(_model, std::nullopt, bolt::python::CtrlCCheck{});
  auto history = trainer.validate_with_data_loader(valid_loader, metrics,
                                                   sparse_inference, verbose);

  return history;
}

auto Mach::predict(const TransformedTable& batch, bool sparse_inference,
                   uint32_t top_k, uint32_t num_scanned_buckets) {
  TransformedTensors tensors(batch);

  auto outputs = _model->forward(tensors.inputs, sparse_inference).at(0);

  uint32_t num_classes = _mach_index->numEntities();
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
    auto predictions = _mach_index->decode(
        /* output = */ vector,
        /* top_k = */ top_k,
        /* num_buckets_to_eval = */ num_scanned_buckets);
    predicted_entities[i] = predictions;
  }

  return predicted_entities;
}

auto predictBuckets(bolt::Model& model, const dataset::mach::MachIndex& index,
                    const bolt::TensorList& inputs,
                    std::optional<uint32_t> num_buckets,
                    bool sparse_inference = false) {
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

auto teach(bolt::ModelPtr& model, TransformedTable train_table,
           TransformedTable balancers, float learning_rate, uint32_t repeats,
           uint32_t epochs, size_t batch_size, const InputMetrics& metrics = {},
           bool verbose = false,
           std::optional<uint32_t> logging_interval = std::nullopt) {
  std::vector<size_t> permutation(train_table.table.numRows() * repeats);
  for (uint32_t i = 0; i < repeats; i++) {
    uint32_t begin = i * train_table.table.numRows();
    uint32_t end = begin + train_table.table.numRows();
    std::iota(permutation.begin() + begin, permutation.begin() + end, 0);
  }
  train_table.table = train_table.table.permute(permutation);

  train_table.removeIntermediateColumns();
  balancers.removeIntermediateColumns();
  train_table.table = train_table.table.concat(balancers.table);

  bolt::Trainer trainer(model);
  return trainer.train(
      /* train_data= */ thirdai::data::toLabeledDataset(
          /* table= */ train_table, /* batch_size= */ batch_size),
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

auto Mach::associate(TransformedTable from_table,
                     const TransformedTable& to_table,
                     TransformedTable balancers, float learning_rate,
                     uint32_t repeats, uint32_t num_buckets, uint32_t epochs,
                     size_t batch_size, const InputMetrics& metrics,
                     bool verbose, std::optional<uint32_t> logging_interval) {
  auto to_tensors = TransformedTensors(to_table).inputs;
  auto to_buckets = predictBuckets(
      /* model= */ *_model, /* index= */ *_mach_index,
      /* inputs= */ to_tensors, /* num_buckets= */ num_buckets);
  auto mach_labels = thirdai::data::ArrayColumn<uint32_t>::make(
      std::move(to_buckets), _mach_index->numBuckets());

  auto doc_id_column = docIdColumn(balancers.labels.value());
  auto dummy_doc_ids = thirdai::data::ValueColumn<uint32_t>::make(
      std::vector<uint32_t>(to_table.table.numRows(), 0),
      std::numeric_limits<uint32_t>::max());

  auto train_table = std::move(from_table);
  train_table.table.setColumn(MACH_LABELS, mach_labels);
  train_table.table.setColumn(doc_id_column, dummy_doc_ids);
  train_table.labels = {thirdai::data::OutputColumns(MACH_LABELS),
                        thirdai::data::OutputColumns(doc_id_column)};

  return teach(_model, std::move(train_table), std::move(balancers),
               learning_rate, repeats, epochs, batch_size, metrics, verbose,
               logging_interval);
}

void Mach::upvote(TransformedTable upvotes, TransformedTable balancers,
                  float learning_rate, uint32_t repeats, uint32_t epochs,
                  size_t batch_size) {
  auto train_table = tableWithMachTransform(std::move(upvotes), _mach_index);
  teach(_model, std::move(train_table), std::move(balancers), learning_rate,
        repeats, epochs, batch_size);
}

auto Mach::outputCorrectness(const TransformedTable& input,
                             const std::vector<uint32_t>& labels,
                             std::optional<uint32_t> num_hashes,
                             bool sparse_inference) {
  auto inputs = TransformedTensors(input).inputs;
  auto top_buckets = predictBuckets(*_model, *_mach_index, inputs, num_hashes,
                                    sparse_inference);

  std::vector<uint32_t> matching_buckets(labels.size());
  std::exception_ptr hashes_err;

#pragma omp parallel for default(none) \
    shared(labels, top_buckets, matching_buckets, hashes_err)
  for (uint32_t i = 0; i < labels.size(); i++) {
    try {
      std::vector<uint32_t> hashes = _mach_index->getHashes(labels[i]);
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

auto Mach::embedding(const TransformedTable& table) {
  // TODO(Nicholas): Sparsity could speed this up, and wouldn't affect the
  // embeddings if the sparsity is in the output layer and the embeddings are
  // from the hidden layer.
  _model->forward(TransformedTensors(table).inputs,
                  /* use_sparsity= */ false);
  return _emb->tensor();
}

auto Mach::entityEmbedding(uint32_t entity) const {
  std::vector<uint32_t> buckets = _mach_index->getHashes(entity);

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

  float index_sparsity = _mach_index->sparsity();
  if (index_sparsity > 0 && index_sparsity <= _mach_sampling_threshold) {
    // TODO(Nicholas) add option to specify new neuron index in set sparsity.
    output_layer->setSparsity(index_sparsity, false, false);
    auto new_index = bolt::MachNeuronIndex::make(_mach_index);
    output_layer->kernel()->setNeuronIndex(new_index);

  } else {
    if (std::dynamic_pointer_cast<bolt::MachNeuronIndex>(neuron_index)) {
      float sparsity = utils::autotuneSparsity(_mach_index->numBuckets());

      auto sampling_config = bolt::DWTASamplingConfig::autotune(
          _mach_index->numBuckets(), sparsity,
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
