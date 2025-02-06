#include "UDTQueryReformulation.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/utils/Timer.h>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/HashFunction.h>
#include <hashing/src/MinHash.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <auto_ml/src/config/FlashConfig.h>
#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/text/Text.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <dataset/src/utils/CsvParser.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/CheckLicense.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <limits>
#include <memory>
#include <stdexcept>

namespace thirdai::automl::udt {

std::optional<std::string> inputColumnName(ColumnDataTypes data_types,
                                           const std::string& target_col) {
  if (data_types.size() != 1 && data_types.size() != 2) {
    throw std::invalid_argument(
        "Only either target or source/target columns must be supplied to "
        "QueryReformulation.");
  }

  data_types.erase(target_col);

  if (data_types.empty()) {
    return std::nullopt;
  }

  if (!asText(data_types.begin()->second)) {
    throw std::invalid_argument(
        "Non target column should be bolt.types.text() in QueryReformulation.");
  }

  return data_types.begin()->first;
}

UDTQueryReformulation::UDTQueryReformulation(
    const ColumnDataTypes& data_types, const std::string& target_column,
    char delimiter, const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args)
    : _flash(std::make_unique<Flash>(inputColumnName(data_types, target_column),
                                     target_column, delimiter, model_config,
                                     user_args)) {}

py::object UDTQueryReformulation::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm, py::kwargs kwargs) {
  (void)learning_rate;
  (void)epochs;
  (void)train_metrics;
  (void)val_data;
  (void)val_metrics;
  (void)callbacks;
  (void)comm;
  (void)kwargs;

  _flash->train(data, options.batch_size, options.verbose);

  return py::none();
}

py::object UDTQueryReformulation::evaluate(
    const dataset::DataSourcePtr& data, const std::vector<std::string>& metrics,
    bool sparse_inference, bool verbose, py::kwargs kwargs) {
  (void)metrics;
  (void)sparse_inference;

  auto top_k = getTopK(kwargs);

  auto results = _flash->evaluate(data, top_k, verbose);
  return py::cast(results);
}

py::object UDTQueryReformulation::predict(const MapInput& sample,
                                          bool sparse_inference,
                                          bool return_predicted_class,
                                          std::optional<uint32_t> top_k,
                                          const py::kwargs& kwargs) {
  (void)sample;
  (void)sparse_inference;
  (void)return_predicted_class;
  (void)top_k;
  return predictBatch({sample}, sparse_inference, return_predicted_class, top_k,
                      kwargs);
}

py::object UDTQueryReformulation::predictBatch(const MapInputBatch& sample,
                                               bool sparse_inference,
                                               bool return_predicted_class,
                                               std::optional<uint32_t> top_k,
                                               const py::kwargs& kwargs) {
  (void)sparse_inference;
  (void)return_predicted_class;
  (void)kwargs;

  auto [phrases, phrase_scores] = _flash->predictBatch(sample, top_k);

  return py::make_tuple(py::cast(phrases), py::cast(phrase_scores));
}

ar::ConstArchivePtr UDTQueryReformulation::toArchive(
    bool with_optimizer) const {
  return _flash->toArchive(with_optimizer);
}

std::unique_ptr<UDTQueryReformulation> UDTQueryReformulation::fromArchive(
    const ar::Archive& archive) {
  return std::make_unique<UDTQueryReformulation>(archive);
}

UDTQueryReformulation::UDTQueryReformulation(const ar::Archive& archive)
    : _flash(Flash::fromArchive(archive)) {}

}  // namespace thirdai::automl::udt
