#include "MachPython.h"
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/Metric.h>
#include <auto_ml/src/udt/Defaults.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/transformations/SpladeAugmentation.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/cold_start/VariableLengthColdStart.h>
#include <dataset/src/DataSource.h>
#include <mach/src/MachConfig.h>
#include <mach/src/MachRetriever.h>
#include <mach/src/MachTrainer.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <limits>

namespace defaults = thirdai::automl::udt::defaults;

namespace thirdai::mach::python {

void loadTrainOptions(const py::kwargs& kwargs, TrainOptions& options) {
  if (kwargs.contains("batch_size")) {
    options.batch_size = kwargs["batch_size"].cast<size_t>();
  }
  if (kwargs.contains("max_in_memory_batches")) {
    options.max_in_memory_batches =
        kwargs["max_in_memory_batches"].cast<std::optional<size_t>>();
  }
  if (kwargs.contains("verbose")) {
    options.verbose = kwargs["verbose"].cast<bool>();
  }
  options.interrupt_check = bolt::python::CtrlCCheck();
}

TrainOptions getTrainOptions(const py::kwargs& kwargs) {
  TrainOptions options;
  loadTrainOptions(kwargs, options);
  return options;
}

ColdStartOptions getColdStartOptions(const py::kwargs& kwargs) {
  ColdStartOptions options;
  loadTrainOptions(kwargs, options);
  if (kwargs.contains("variable_length")) {
    options.variable_length =
        kwargs["variable_length"].cast<data::VariableLengthConfig>();
  }
  if (kwargs.contains("splade_config")) {
    options.splade_config = kwargs["splade_config"].cast<data::SpladeConfig>();
  }
  return options;
}

EvaluateOptions getEvaluateOptions(const py::kwargs& kwargs) {
  EvaluateOptions options;
  if (kwargs.contains("verbose")) {
    options.verbose = kwargs["verbose"].cast<bool>();
  }
  if (kwargs.contains("use_sparsity")) {
    options.use_sparsity = kwargs["use_sparsity"].cast<bool>();
  }
  return options;
}

char getCsvDelimiter(const py::kwargs& kwargs) {
  if (kwargs.contains("delimiter")) {
    return kwargs["delimiter"].cast<char>();
  }
  return ',';
}

char getLabelDelimiter(const py::kwargs& kwargs) {
  if (kwargs.contains("label_delimiter")) {
    return kwargs["label_delimiter"].cast<char>();
  }
  return ':';
}

data::ColumnMapIteratorPtr csvIterator(const std::string& filename,
                                       const std::string& id_col,
                                       const py::kwargs& kwargs) {
  return data::TransformedIterator::make(
      data::CsvIterator::make(dataset::FileDataSource::make(filename),
                              getCsvDelimiter(kwargs)),
      std::make_shared<data::StringToTokenArray>(
          id_col, id_col, getLabelDelimiter(kwargs),
          std::numeric_limits<uint32_t>::max()),
      nullptr);
}

bolt::metrics::History wrappedTrain(
    const MachRetrieverPtr& mach, const data::ColumnMapIteratorPtr& data,
    float learning_rate, uint32_t epochs,
    const std::vector<std::string>& metrics,
    const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
    const py::kwargs& kwargs) {
  auto options = getTrainOptions(kwargs);

  return mach->train(data, learning_rate, epochs, metrics, callbacks, options);
}

bolt::metrics::History wrappedTrainOnCsv(
    const MachRetrieverPtr& mach, const std::string& filename,
    float learning_rate, uint32_t epochs,
    const std::vector<std::string>& metrics,
    const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
    const py::kwargs& kwargs) {
  auto data = csvIterator(filename, mach->idCol(), kwargs);
  auto options = getTrainOptions(kwargs);

  return mach->train(data, learning_rate, epochs, metrics, callbacks, options);
}

bolt::metrics::History wrappedColdStart(
    const MachRetrieverPtr& mach, const data::ColumnMapIteratorPtr& data,
    const std::vector<std::string>& strong_cols,
    const std::vector<std::string>& weak_cols, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
    const py::kwargs& kwargs) {
  auto coldstart_options = getColdStartOptions(kwargs);

  return mach->coldstart(data, strong_cols, weak_cols, learning_rate, epochs,
                         metrics, callbacks, coldstart_options);
}

bolt::metrics::History wrappedColdStartOnCsv(
    const MachRetrieverPtr& mach, const std::string& filename,
    const std::vector<std::string>& strong_cols,
    const std::vector<std::string>& weak_cols, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
    const py::kwargs& kwargs) {
  auto data = csvIterator(filename, mach->idCol(), kwargs);
  auto coldstart_options = getColdStartOptions(kwargs);

  return mach->coldstart(data, strong_cols, weak_cols, learning_rate, epochs,
                         metrics, callbacks, coldstart_options);
}

bolt::metrics::History wrappedEvaluate(const MachRetrieverPtr& mach,
                                       const data::ColumnMapIteratorPtr& data,
                                       const std::vector<std::string>& metrics,
                                       const py::kwargs& kwargs) {
  return mach->evaluate(data, metrics, getEvaluateOptions(kwargs));
}

bolt::metrics::History wrappedEvaluateOnCsv(
    const MachRetrieverPtr& mach, const std::string& filename,
    const std::vector<std::string>& metrics, const py::kwargs& kwargs) {
  auto data = csvIterator(filename, mach->idCol(), kwargs);

  return mach->evaluate(data, metrics, getEvaluateOptions(kwargs));
}

void defineMach(py::module_& module) {
  py::class_<MachConfig>(module, "MachConfig")
      .def(py::init<>())
      .def("build", &MachConfig::build)
      .def("text_col", &MachConfig::textCol, py::arg("col"))
      .def("id_col", &MachConfig::idCol, py::arg("col"))
      .def("tokenizer", &MachConfig::tokenizer, py::arg("tokenizer"))
      .def("contextual_encoding", &MachConfig::contextualEncoding,
           py::arg("encoding"))
      .def("lowercase", &MachConfig::lowercase, py::arg("lowercase") = true)
      .def("text_feature_dim", &MachConfig::textFeatureDim,
           py::arg("text_feature_dim"))
      .def("emb_dim", &MachConfig::embDim, py::arg("emb_dim"))
      .def("n_buckets", &MachConfig::nBuckets, py::arg("n_bukcets"))
      .def("emb_bias", &MachConfig::embBias, py::arg("bias") = true)
      .def("output_bias", &MachConfig::outputBias, py::arg("bias") = true)
      .def("emb_activation", &MachConfig::embActivation, py::arg("activation"))
      .def("output_activation", &MachConfig::outputActivation,
           py::arg("activation"))
      .def("n_hashes", &MachConfig::nHashes, py::arg("n_hashes"))
      .def("mach_sampling_threshold", &MachConfig::machSamplingThreshold,
           py::arg("threshold"))
      .def("num_buckets_to_eval", &MachConfig::nBucketsToEval,
           py::arg("n_buckets_to_eval"))
      .def("mach_memory_params", &MachConfig::machMemoryParams,
           py::arg("max_memory_ids"), py::arg("max_memory_samples_per_id"))
      .def("freeze_hash_tables_epoch", &MachConfig::freezeHashTablesEpoch,
           py::arg("epoch"));

  py::class_<MachRetriever, MachRetrieverPtr>(module, "MachRetriever")
#if THIRDAI_EXPOSE_ALL
      .def_property_readonly("model", &MachRetriever::model)
#endif
      .def_property_readonly("index", &MachRetriever::index)
      .def("train", &wrappedTrain, py::arg("data"), py::arg("learning_rate"),
           py::arg("epochs"), py::arg("metrics") = std::vector<std::string>{},
           py::arg("callbacks") = std::vector<bolt::callbacks::CallbackPtr>{})
      .def("train", &wrappedTrainOnCsv, py::arg("filename"),
           py::arg("learning_rate"), py::arg("epochs"),
           py::arg("metrics") = std::vector<std::string>{},
           py::arg("callbacks") = std::vector<bolt::callbacks::CallbackPtr>{})
      .def("coldstart", &wrappedColdStart, py::arg("data"),
           py::arg("strong_cols"), py::arg("weak_cols"),
           py::arg("learning_rate"), py::arg("epochs"),
           py::arg("metrics") = std::vector<std::string>{},
           py::arg("callbacks") = std::vector<bolt::callbacks::CallbackPtr>{})
      .def("coldstart", &wrappedColdStartOnCsv, py::arg("filename"),
           py::arg("strong_cols"), py::arg("weak_cols"),
           py::arg("learning_rate"), py::arg("epochs"),
           py::arg("metrics") = std::vector<std::string>{},
           py::arg("callbacks") = std::vector<bolt::callbacks::CallbackPtr>{})
      .def("evaluate", &wrappedEvaluate, py::arg("data"), py::arg("metrics"))
      .def("evaluate", &wrappedEvaluateOnCsv, py::arg("filename"),
           py::arg("metrics"))
      .def("search", &MachRetriever::search, py::arg("queries"),
           py::arg("top_k"), py::arg("sparse_inference") = false)
      .def("search", &MachRetriever::searchSingle, py::arg("query"),
           py::arg("top_k"), py::arg("sparse_inference") = false)
      .def("search", &MachRetriever::searchBatch, py::arg("queries"),
           py::arg("top_k"), py::arg("sparse_inference") = false)
      .def("rank", &MachRetriever::rank, py::arg("queries"),
           py::arg("candidates"), py::arg("top_k"),
           py::arg("sparse_inference") = false)
      .def("rank", &MachRetriever::rankSingle, py::arg("query"),
           py::arg("candidates"), py::arg("top_k"),
           py::arg("sparse_inference") = false)
      .def("rank", &MachRetriever::rankBatch, py::arg("queries"),
           py::arg("candidates"), py::arg("top_k"),
           py::arg("sparse_inference") = false)
      .def("introduce", &MachRetriever::introduceIterator, py::arg("data"),
           py::arg("strong_cols"), py::arg("weak_cols"),
           py::arg("text_augmentation") = true,
           py::arg("n_buckets_to_sample") = std::nullopt,
           py::arg("n_random_hashes") = 0, py::arg("load_balancing") = true,
           py::arg("sort_random_hashes") = false)
      .def("erase", &MachRetriever::erase, py::arg("ids"))
      .def("clear", &MachRetriever::clear)
      .def("upvote", &MachRetriever::upvote, py::arg("upvotes"),
           py::arg("n_upvote_samples") = defaults::RLHF_N_FEEDBACK_SAMPLES,
           py::arg("n_balancing_samples") = defaults::RLHF_N_BALANCING_SAMPLES,
           py::arg("learning_rate") = defaults::RLHF_LEARNING_RATE,
           py::arg("epochs") = defaults::RLHF_EPOCHS,
           py::arg("batch_size") = defaults::RLHF_BATCH_SIZE)
      .def("upvote", &MachRetriever::upvoteBatch, py::arg("queries"),
           py::arg("ids"),
           py::arg("n_upvote_samples") = defaults::RLHF_N_FEEDBACK_SAMPLES,
           py::arg("n_balancing_samples") = defaults::RLHF_N_BALANCING_SAMPLES,
           py::arg("learning_rate") = defaults::RLHF_LEARNING_RATE,
           py::arg("epochs") = defaults::RLHF_EPOCHS,
           py::arg("batch_size") = defaults::RLHF_BATCH_SIZE)
      .def("associate", &MachRetriever::associate, py::arg("sources"),
           py::arg("targets"), py::arg("n_buckets"),
           py::arg("n_association_samples") = defaults::RLHF_N_FEEDBACK_SAMPLES,
           py::arg("n_balancing_samples") = defaults::RLHF_N_BALANCING_SAMPLES,
           py::arg("learning_rate") = defaults::RLHF_LEARNING_RATE,
           py::arg("epochs") = defaults::RLHF_EPOCHS,
           py::arg("force_non_empty") = true,
           py::arg("batch_size") = defaults::RLHF_BATCH_SIZE)
      .def("associate", &MachRetriever::associateBatch, py::arg("sources"),
           py::arg("targets"), py::arg("n_buckets"),
           py::arg("n_association_samples") = defaults::RLHF_N_FEEDBACK_SAMPLES,
           py::arg("n_balancing_samples") = defaults::RLHF_N_BALANCING_SAMPLES,
           py::arg("learning_rate") = defaults::RLHF_LEARNING_RATE,
           py::arg("epochs") = defaults::RLHF_EPOCHS,
           py::arg("force_non_empty") = true,
           py::arg("batch_size") = defaults::RLHF_BATCH_SIZE)
      .def("save", &MachRetriever::save, py::arg("filename"),
           py::arg("with_optimizer") = false)
      .def_static("load", &MachRetriever::load, py::arg("filename"));

  py::class_<DataCheckpoint>(module, "DataCheckpoint")
      .def(py::init<data::ColumnMapIteratorPtr, std::string,
                    std::vector<std::string>>(),
           py::arg("data_iter"), py::arg("id_col"), py::arg("text_cols"));

  py::class_<MachTrainer>(module, "MachTrainer")
      .def(py::init<MachRetrieverPtr, DataCheckpoint>(), py::arg("model"),
           py::arg("data"))
      .def("complete", &MachTrainer::complete, py::arg("ckpt_dir"))
      .def_static("from_checkpoint", &MachTrainer::fromCheckpoint,
                  py::arg("ckpt_dir"))
      .def("strong_weak_cols", &MachTrainer::strongWeakCols,
           py::arg("strong_cols"), py::arg("weak_cols"))
      .def("vlc", &MachTrainer::vlc, py::arg("vlc"))
      .def("learning_rate", &MachTrainer::learningRate,
           py::arg("learning_rate"))
      .def("min_max_epochs", &MachTrainer::minMaxEpochs, py::arg("min_epochs"),
           py::arg("max_epochs"))
      .def("metrics", &MachTrainer::metrics, py::arg("metrics"))
      .def("max_in_memory_batches", &MachTrainer::maxInMemoryBatches,
           py::arg("max_in_memory_batches"))
      .def("batch_size", &MachTrainer::batchSize, py::arg("batch_size"))
      .def("early_stop", &MachTrainer::earlyStop, py::arg("metric"),
           py::arg("threshold"));
}

}  // namespace thirdai::mach::python