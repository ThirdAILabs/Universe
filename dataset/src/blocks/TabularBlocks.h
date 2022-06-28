#pragma once

#include "BlockInterface.h"
#include <dataset/src/bolt_datasets/batch_processors/PairgramHasher.h>
#include <unordered_map>

namespace thirdai::dataset {

enum class TabularDataType {
  Numeric,
  Categorical,
  Label
};  // TODO(david) add datetime/text support

// TODO(david) verify each column is valid?
class TabularMetadata {
 public:
  TabularMetadata() {}

  // TODO(david) use a different map here?
  uint32_t numColumns() const { return _col_to_type.size(); }

  TabularDataType getType(uint32_t col) { return _col_to_type[col]; }

  uint32_t numClasses() const { return _class_id_to_class.size(); }

  std::vector<std::string> getColumnNames() { return _column_names; }

  // TODO(david) check if invalid id
  std::string getClassName(uint32_t class_id) {
    return _class_id_to_class[class_id];
  }
  // TODO(david) check if invalid/new label?? or could do this in parsing
  uint32_t getClassId(const std::vector<std::string_view>& input_row) {
    _class_to_class_id[input_row[label_col_index]];
  }

 private:
  uint32_t label_col_index;
  std::vector<std::string> _column_names;
  std::unordered_map<uint32_t, TabularDataType> _col_to_type;
  std::unordered_map<std::string_view, uint32_t> _class_to_class_id;
  std::vector<std::string> _class_id_to_class;
};

class TabularPairGram : public Block {
 public:
  TabularPairGram(TabularMetadata& metadata, uint32_t output_range)
      : _metadata(metadata), _output_range(output_range) {}

  uint32_t featureDim() const final { return _output_range; };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final { return _metadata.numColumns(); };

 protected:
  // TODO(david) calculate unigrams and capped pairgrams correctly
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    std::vector<uint32_t> unigram_hashes;
    for (uint32_t col = 0; col < input_row.size(); col++) {
      uint32_t unigram;
      switch (_metadata.getType(col)) {
        case TabularDataType::Numeric:
          // bin the column according to min and max
          // get the unigram of the column
          input_row[col];
        case TabularDataType::Categorical:
          // get unigram from the column
          input_row[col];
      }
      unigram_hashes.push_back(unigram);
      vec.addSparseFeatureToSegment(unigram % _output_range,
                                    1.0);  // TODO(david) update
    }

    // at the minimum must have unigrams
    // if we have less than 1000 unigrams, add pairgrams of random columns (same
    // random columns for each row) up to a cap of 1000
  }

 private:
  TabularMetadata _metadata;
  uint32_t _output_range;
};

class TabularLabel : public Block {
 public:
  explicit TabularLabel(TabularMetadata& metadata) : _metadata(metadata) {}

  uint32_t featureDim() const final { return _metadata.numClasses(); };

  bool isDense() const final { return true; };

  uint32_t expectedNumColumns() const final { return 1; };

 protected:
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    vec.addSparseFeatureToSegment(_metadata.getClassId(input_row), 1.0);
  }

 private:
  TabularMetadata _metadata;
};

}  // namespace thirdai::dataset
