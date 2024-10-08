#pragma once

#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <string>
#include <vector>

namespace thirdai::data {

class StringSplitOnWhiteSpace final : public Transformation {
 public:
  /**
   * Note: The unicode version returns offsets using the unicode characters,
   * this means that the offsets will not be correct if applied to the string
   * in ascii format.
   */
  StringSplitOnWhiteSpace(std::string input_column_name,
                          std::string output_column_name,
                          bool as_unicode = false)
      : _input_column_name(std::move(input_column_name)),
        _output_column_name(std::move(output_column_name)),
        _as_unicode(as_unicode) {}

  explicit StringSplitOnWhiteSpace(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "string_split_whitespace"; }

 private:
  std::string _input_column_name;
  std::string _output_column_name;
  bool _as_unicode;
};

}  // namespace thirdai::data
