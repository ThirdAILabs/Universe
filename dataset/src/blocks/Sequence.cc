#include "Sequence.h"
#include <cereal/archives/binary.hpp>
#include <hashing/src/HashUtils.h>
#include <dataset/src/featurizers/ProcessorUtils.h>
#include <dataset/src/utils/CsvParser.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/text/StringManipulation.h>
#include <limits>
#include <memory>
#include <stdexcept>

namespace thirdai::dataset {

SequenceBlock::SequenceBlock(ColumnIdentifier col, char delimiter, uint32_t dim)
    : _col(std::move(col)), _delimiter(delimiter), _dim(dim) {}

uint32_t SequenceBlock::featureDim() const { return _dim; }

bool SequenceBlock::isDense() const { return false; }

Explanation SequenceBlock::explainIndex(uint32_t index_within_block,
                                        ColumnarInputSample& input) {
  std::string keyword;

  std::string sequence(input.column(_col));
  auto elements = parsers::CSV::parseLine(sequence, _delimiter);
  for (uint32_t i = 0; i < elements.size(); i++) {
    if (sequenceHash(elements[i], /* pos= */ i) == index_within_block) {
      keyword = std::string(elements[i]);
    }
  }

  return {_col, std::move(keyword)};
}

void SequenceBlock::buildSegment(ColumnarInputSample& input,
                                 SegmentedFeatureVector& vec) {
  auto sequence = input.column(_col);
  auto elements = parsers::CSV::parseLine(sequence, _delimiter);
  std::vector<uint32_t> hashes(elements.size());
  for (uint32_t i = 0; i < elements.size(); i++) {
    hashes[i] = sequenceHash(elements[i], /* pos= */ i);
  }

  for (auto& [index, value] : token_encoding::sumRepeatedIndices(hashes)) {
    vec.addSparseFeatureToSegment(index, value);
  }
}

std::vector<ColumnIdentifier*> SequenceBlock::concreteBlockColumnIdentifiers() {
  return {&_col};
}

uint32_t SequenceBlock::sequenceHash(std::string_view element,
                                     uint32_t pos) const {
  auto element_hash =
      token_encoding::seededMurmurHash(element.data(), element.size());
  return hashing::combineHashes(pos, element_hash) % _dim;
}

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::SequenceBlock)