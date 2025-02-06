#pragma once

#include <archive/src/Archive.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/backends/cpp/Flash.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <search/src/Flash.h>
#include <memory>
#include <optional>
#include <unordered_map>

namespace thirdai::automl::udt {

class UDTQueryReformulation final : public UDTBackend {
 public:
  UDTQueryReformulation(const ColumnDataTypes& data_types,
                        const std::string& target_column, char delimiter,
                        const std::optional<std::string>& model_config,
                        const config::ArgumentMap& user_args);

  explicit UDTQueryReformulation(const ar::Archive& archive);

  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions option, const bolt::DistributedCommPtr& comm,
                   py::kwargs kwargs) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose,
                      py::kwargs kwargs) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class, std::optional<uint32_t> top_k,
                     const py::kwargs& kwargs) final;

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k,
                          const py::kwargs& kwargs) final;

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::unique_ptr<UDTQueryReformulation> fromArchive(
      const ar::Archive& archive);

  static std::string type() { return Flash::type(); }

 private:
  static uint32_t getTopK(const py::kwargs& kwargs) {
    if (!kwargs.contains("top_k")) {
      throw std::invalid_argument(
          "top_k is a required argument for query reformulation.");
    }
    return kwargs["top_k"].cast<uint32_t>();
  }

  std::unique_ptr<Flash> _flash;
};

}  // namespace thirdai::automl::udt