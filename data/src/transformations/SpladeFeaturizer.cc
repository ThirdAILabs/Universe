#include "SpladeFeaturizer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <utility>

namespace thirdai::data {

SpladeFeaturizer::SpladeFeaturizer(uint32_t context_length,
                                   bool fill_empty_contexts,
                                   std::string source_column,
                                   uint32_t partition_length,
                                   std::string output_interval_prefix)
    : _context_length(context_length),
      _fill_empty_contexts(fill_empty_contexts),
      _source_column(std::move(source_column)),
      _partition_length(partition_length),
      _output_interval_prefix(std::move(output_interval_prefix)) {
  if (context_length % partition_length != 0) {
    throw std::logic_error("Context length with value of " +
                           std::to_string(context_length) +
                           " couldn't be divided with partition length of " +
                           std::to_string(partition_length));
  }
}

ColumnMap SpladeFeaturizer::apply(ColumnMap columns, State& state) const {
  (void)state;
  auto texts = columns.getArrayColumn<uint32_t>(_source_column);
  uint32_t num_inputs = _context_length / _partition_length;
  std::vector<std::vector<std::vector<uint32_t>>> intervals(num_inputs);
  for (size_t i = 0; i < num_inputs; i++) {
    intervals.at(i).assign(texts->numRows(), {});
  }

  std::exception_ptr error;
#pragma omp parallel for default(none) \
    shared(texts, intervals, error, num_inputs)
  for (size_t i = 0; i < texts->numRows(); i++) {
    try {
      auto input_tokens = texts->row(i);

      std::vector<uint32_t> tokens;

      tokens.insert(tokens.end(), input_tokens.begin(), input_tokens.end());
      uint32_t tokens_size = tokens.size();
      uint32_t current_partition = 0;
      for(uint32_t start =0; start < tokens_size; start += _partition_length){
        intervals[current_partition][i] = {
            tokens.begin() + start, tokens.begin() + std::min(tokens_size, start + _partition_length)
        };
        current_partition += 1;
      }
        uint32_t num_buckets_filled = std::ceil(std::log2( tokens_size+ 1));

        uint32_t num_input_tokens_to_fill = std::pow(2, num_buckets_filled);

      if(_fill_empty_contexts && num_input_tokens_to_fill < _context_length){
        uint32_t set_of_buckets_filled = 1;
        while(num_input_tokens_to_fill * set_of_buckets_filled < _context_length){
            uint32_t current_partition = num_buckets_filled * set_of_buckets_filled;
            for(uint32_t start =0; start < tokens_size; start += _partition_length){
                intervals[current_partition][i] = {
            tokens.begin() + start, tokens.begin() + std::min(tokens_size, start + _partition_length)
                };
                current_partition += 1;
            }
            set_of_buckets_filled += 1;
        }
      }

    } catch (...) {
#pragma omp critical
      error = std::current_exception();
    }
  }
  if (error) {
    std::rethrow_exception(error);
  }
  std::unordered_map<std::string, ColumnPtr> output_columns;

  for (size_t interval = 0; interval < num_inputs; interval++) {
    std::string name =
        _output_interval_prefix + std::to_string(1 << interval);

    output_columns[name] = ArrayColumn<uint32_t>::make(
        std::move(intervals[interval]), texts->dim());
  }

  return ColumnMap(output_columns);
}

ar::ConstArchivePtr SpladeFeaturizer::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  map->set("fill_empty_contexts", ar::boolean(_fill_empty_contexts));
  map->set("context_length", ar::u64(_context_length));

  return map;
}

SpladeFeaturizer::SpladeFeaturizer(const ar::Archive& archive)
    : _context_length(archive.u64("context_length")),
      _fill_empty_contexts(archive.getAs<ar::Boolean>("fill_empty_contexts")) {}

template void SpladeFeaturizer::serialize(cereal::BinaryInputArchive&);
template void SpladeFeaturizer::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void SpladeFeaturizer::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _context_length,
          _fill_empty_contexts);
}

}  // namespace thirdai::data