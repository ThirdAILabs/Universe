#pragma once

#include <archive/src/Archive.h>
#include <auto_ml/src/pretrained/SpladeMach.h>
#include <data/src/transformations/Transformation.h>
#include <regex>
#include <stdexcept>

namespace thirdai::data {

class SpladeMachAugmentation final : public Transformation {
 public:
  SpladeMachAugmentation(std::string input_column, std::string output_column,
                         std::shared_ptr<automl::SpladeMach> model,
                         size_t n_hashes_per_model)
      : _input_column(std::move(input_column)),
        _output_column(std::move(output_column)),
        _model(std::move(model)),
        _n_hashes_per_model(n_hashes_per_model) {}

  explicit SpladeMachAugmentation(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "splade_mach_augmentation"; }

 private:
  std::string _input_column;
  std::string _output_column;

  std::shared_ptr<automl::SpladeMach> _model;
  size_t _n_hashes_per_model;
};

}  // namespace thirdai::data