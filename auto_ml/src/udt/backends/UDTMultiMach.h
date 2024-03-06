#pragma once

#include <bolt/src/nn/model/Model.h>
#include <archive/src/Archive.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/backends/UDTMach.h>
#include <stdexcept>

namespace thirdai::automl::udt {

using Scores = std::vector<std::pair<uint32_t, float>>;

struct BestScore {
  bool operator()(const std::pair<uint32_t, float>& a,
                  const std::pair<uint32_t, float>& b) {
    return a.second > b.second;
  }
};

class UDTMultiMach final : public UDTBackend {
 public:
  UDTMultiMach(
      const ColumnDataTypes& input_data_types,
      const UserProvidedTemporalRelationships& temporal_tracking_relationships,
      const std::string& target_name, const CategoricalDataTypePtr& target,
      uint32_t n_target_classes, bool integer_target,
      const TabularOptions& tabular_options,
      const std::optional<std::string>& model_config,
      config::ArgumentMap user_args);

  explicit UDTMultiMach(const ar::Archive& archive);

  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options,
                   const bolt::DistributedCommPtr& comm) final;

  py::object coldstart(
      const dataset::DataSourcePtr& data,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      std::optional<data::VariableLengthConfig> variable_length,
      float learning_rate, uint32_t epochs,
      const std::vector<std::string>& train_metrics,
      const dataset::DataSourcePtr& val_data,
      const std::vector<std::string>& val_metrics,
      const std::vector<CallbackPtr>& callbacks, TrainOptions options,
      const bolt::DistributedCommPtr& comm) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose,
                      std::optional<uint32_t> top_k) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class,
                     std::optional<uint32_t> top_k) final;

  py::object predictBatch(const MapInputBatch& samples, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k) final;

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::unique_ptr<UDTMultiMach> fromArchive(const ar::Archive& archive);

  void setDecodeParams(uint32_t top_k_to_return,
                       uint32_t num_buckets_to_eval) final {
    _default_top_k_to_return = top_k_to_return;
    _num_buckets_to_eval = num_buckets_to_eval;
  }

  static std::string type() { return "udt_multi_mach"; }

  void enableFastDecode() { _fast_decode = true; }

  void disableFastDecode() { _fast_decode = false; }

  bolt::ModelPtr model() const final { return _models[0]->model(); }

 private:
  std::vector<Scores> predictImpl(const MapInputBatch& input,
                                  bool sparse_inference, uint32_t top_k);

  std::vector<Scores> predictFastDecode(const MapInputBatch& input,
                                        bool sparse_inference, uint32_t top_k);

  std::vector<Scores> predictRegularDecode(const MapInputBatch& input,
                                           bool sparse_inference,
                                           uint32_t top_k);

  std::vector<std::unique_ptr<UDTMach>> _models;

  uint32_t _default_top_k_to_return = defaults::MACH_TOP_K_TO_RETURN;
  uint32_t _num_buckets_to_eval = defaults::MACH_NUM_BUCKETS_TO_EVAL;

  bool _fast_decode = false;
};

}  // namespace thirdai::automl::udt