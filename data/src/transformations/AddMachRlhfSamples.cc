#include "AddMachRlhfSamples.h"
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <cstdint>

namespace thirdai::data {

AddMachRlhfSamples::AddMachRlhfSamples(std::string input_indices_column,
                                       std::string input_values_column,
                                       std::string label_column,
                                       std::string mach_buckets_column)

    : _input_indices_column(std::move(input_indices_column)),
      _input_values_column(std::move(input_values_column)),
      _label_column(std::move(label_column)),
      _mach_buckets_column(std::move(mach_buckets_column)) {}

ColumnMap AddMachRlhfSamples::apply(ColumnMap columns, State& state) const {
  const auto& labels = columns.getArrayColumn<uint32_t>(_label_column);
  const auto& input_indices =
      columns.getArrayColumn<uint32_t>(_input_indices_column);
  const auto& input_values =
      columns.getArrayColumn<float>(_input_values_column);
  const auto& mach_buckets =
      columns.getArrayColumn<uint32_t>(_mach_buckets_column);
  for (size_t i = 0; i < columns.numRows(); i++) {
    if (labels->row(i).size() < 1) {
      continue;
    }
    uint32_t doc_id = labels->row(i)[0];
    automl::udt::RlhfSample sample;
    sample.input_indices = {input_indices->row(i).begin(),
                            input_indices->row(i).end()};
    sample.input_values = {input_values->row(i).begin(),
                           input_values->row(i).end()};
    sample.mach_buckets = {mach_buckets->row(i).begin(),
                           mach_buckets->row(i).end()};
    state.rlhfSampler().addSample(doc_id, std::move(sample));
  }

  return columns;
}

template void AddMachRlhfSamples::serialize(cereal::BinaryInputArchive&);
template void AddMachRlhfSamples::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void AddMachRlhfSamples::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_indices_column,
          _input_values_column, _label_column, _mach_buckets_column);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::AddMachRlhfSamples)