#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class SpladeFeaturizer final : public Transformation {
 public:
  SpladeFeaturizer(uint32_t context_length, bool fill_empty_contexts,
                   std::string source_column, uint32_t partition_length,
                   std::string output_interval_prefix);

  explicit SpladeFeaturizer(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "splade_featurizer"; }

 private:
  uint32_t _context_length;
  bool _fill_empty_contexts;
  std::string _source_column;
  uint32_t _partition_length;
  std::string _output_interval_prefix;

  SpladeFeaturizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data