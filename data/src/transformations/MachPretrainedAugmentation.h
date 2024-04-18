#pragma once

#include <archive/src/Archive.h>
#include <auto_ml/src/pretrained/MachPretrained.h>
#include <data/src/transformations/Transformation.h>
#include <regex>
#include <stdexcept>

namespace thirdai::data {

class MachPretrainedAugmentation final : public Transformation {
 public:
  MachPretrainedAugmentation(std::string input_column,
                             std::string output_column,
                             std::shared_ptr<automl::MachPretrained> model,
                             size_t n_hashes_per_model)
      : _input_column(std::move(input_column)),
        _output_column(std::move(output_column)),
        _model(std::move(model)),
        _n_hashes_per_model(n_hashes_per_model) {}

  // explicit MachPretrainedAugmentation(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final {
    throw std::invalid_argument("TODO(Nicholas): toArchive");
  }

 private:
  std::string _input_column;
  std::string _output_column;

  std::shared_ptr<automl::MachPretrained> _model;
  size_t _n_hashes_per_model;
};

}  // namespace thirdai::data