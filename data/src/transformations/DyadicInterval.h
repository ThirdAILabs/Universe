#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class DyadicInterval final : public Transformation {
 public:
  DyadicInterval(std::string input_column,
                 std::optional<std::string> prompt_column,
                 std::string output_interval_prefix, std::string target_column,
                 std::optional<std::string> context_length_column,
                 size_t n_intervals, bool is_bidirectional = false);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ColumnMap inferenceFeaturization(ColumnMap columns) const;

 private:
  static std::vector<size_t> computeOffsets(
      const ArrayColumnBasePtr<uint32_t>& texts, size_t chunk_size);

  std::optional<std::string> _prompt_column;
  std::string _input_column;
  std::string _output_interval_prefix;
  std::string _target_column;
  std::optional<std::string> _context_length_column;

  bool _is_bidirectional;

  size_t _n_intervals;

  DyadicInterval() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data