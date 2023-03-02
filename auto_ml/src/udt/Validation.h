#pragma once

#include <dataset/src/DataSource.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>

namespace thirdai::automl::udt {

/**
 * Stores necessary information for validation, used to simplify args and avoid
 * having to pass in multiple optional arguments and verify that they are
 * correctly specified together.
 */
class ValidationArgs {
 public:
  explicit ValidationArgs(
      std::vector<std::string> metrics,
      std::optional<uint32_t> steps_per_validation = std::nullopt,
      bool sparse_inference = false)
      : _metrics(std::move(metrics)),
        _steps_per_validation(steps_per_validation),
        _sparse_inference(sparse_inference) {}

  const auto& metrics() const { return _metrics; }

  const auto& stepsPerValidation() const { return _steps_per_validation; }

  bool sparseInference() const { return _sparse_inference; }

 private:
  std::vector<std::string> _metrics;
  std::optional<uint32_t> _steps_per_validation;
  bool _sparse_inference;
};

template <typename T>
class Validation {
 public:
  explicit Validation(T data, ValidationArgs args)
      : _data(std::move(data)), _args(std::move(args)) {}

  const T& data() const { return _data; }

  const ValidationArgs& args() const { return _args; }

 private:
  T _data;
  ValidationArgs _args;
};

using DataSourceValidation = Validation<dataset::DataSourcePtr>;
using DatasetLoaderValidation = Validation<dataset::DatasetLoaderPtr>;

}  // namespace thirdai::automl::udt