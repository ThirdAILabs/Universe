#pragma once

#include <data/src/transformations/Transformation.h>
#include <cstddef>
#include <string>

namespace thirdai::data {

/**
 * Hashes tokens according to their position in a sequence of tokens.
 * For example, given a row of tokens [a, a, a], this will create a new row
 * of unique tokens [a1, a2, a3] since the same token is hashed differently
 * depending on its position. Note that they are not guaranteed to be unique
 * since hash collisions may happen.
 */
class HashPositionTransform final : public Transformation {
 public:
  HashPositionTransform(std::string input_column, std::string output_column,
                        size_t hash_range)
      : _input_column(std::move(input_column)),
        _output_column(std::move(output_column)),
        _dim(hash_range) {}

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  ar::ConstArchivePtr toArchive() const final;

 private:
  std::string _input_column;
  std::string _output_column;
  size_t _dim;

  HashPositionTransform() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

/**
 * Offsets each token id by position * vocab size (assumed to be the dimension
 * of the input column).
 * Note that if a token is in position >= max_num_tokens, it will be encoded as
 * if it is in position max_num_tokens - 1.
 *
 * For example, given a row of tokens [3, 2, 1] in a token array column of
 * dimension 5 and max_num_tokens = 2, this produces a row [3, 7, 6].
 * 3 = position 0 * vocab size 5 + 3 = 3
 * 7 = position 1 * vocab size 5 + 2 = 7
 * 3 = position 1 * vocab size 5 + 1 = 6
 */
class OffsetPositionTransform final : public Transformation {
 public:
  OffsetPositionTransform(std::string input_column, std::string output_column,
                          size_t max_num_tokens)
      : _input_column(std::move(input_column)),
        _output_column(std::move(output_column)),
        _max_num_tokens(max_num_tokens) {}

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "encode_position"; }

 private:
  std::string _input_column;
  std::string _output_column;
  size_t _max_num_tokens;

  OffsetPositionTransform() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data