#include "Sequence.h"
#include <cereal/archives/binary.hpp>
#include <hashing/src/HashUtils.h>
#include <dataset/src/featurizers/ProcessorUtils.h>
#include <dataset/src/utils/CsvParser.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/StringManipulation.h>
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

  auto sequence = std::string(input.column(_col));
  auto elements = parsers::CSV::parseLine(sequence, _delimiter);
  for (uint32_t i = 0; i < elements.size(); i++) {
    if (sequenceHash(elements[i], /* pos= */ i) == index_within_block) {
      keyword = std::string(elements[i]);
    }
  }

  return {_col, std::move(keyword)};
}

std::exception_ptr SequenceBlock::buildSegment(ColumnarInputSample& input,
                                               SegmentedFeatureVector& vec) {
  auto sequence = std::string(input.column(_col));
  auto elements = parsers::CSV::parseLine(sequence, _delimiter);
  std::vector<uint32_t> hashes(elements.size());
  for (uint32_t i = 0; i < elements.size(); i++) {
    hashes[i] = sequenceHash(elements[i], /* pos= */ i);
  }

  for (auto& [index, value] : token_encoding::sumRepeatedIndices(hashes)) {
    vec.addSparseFeatureToSegment(index, value);
  }
  return nullptr;
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

SequenceTargetBlock::SequenceTargetBlock(ColumnIdentifier target_col,
                                         ColumnIdentifier step_col,
                                         uint32_t max_steps,
                                         uint32_t vocabulary_size)
    : _target_col(std::move(target_col)),
      _step_col(std::move(step_col)),
      _max_steps(max_steps),
      _vocabulary(vocabulary_size, /* limit_vocab_size= */ true),
      _vocabulary_size(vocabulary_size) {}

void SequenceTargetBlock::prepareForBatch(ColumnarInputBatch& incoming_batch) {
  (void)incoming_batch;
  if (_vocabulary.vocabSize() == _vocabulary_size) {
    _vocabulary.fixVocab();
  }
}

uint32_t SequenceTargetBlock::featureDim() const {
  return _vocabulary.vocabSize() * _max_steps;
}

bool SequenceTargetBlock::isDense() const { return false; }

Explanation SequenceTargetBlock::explainIndex(uint32_t index_within_block,
                                              ColumnarInputSample& input) {
  (void)index_within_block;
  (void)input;
  throw std::logic_error(
      "Sequence target block can only be used to create label vectors so "
      "explainIndex() is not supported.");
}

std::exception_ptr SequenceTargetBlock::buildSegment(
    ColumnarInputSample& input, SegmentedFeatureVector& vec) {
  auto target = std::string(input.column(_target_col));
  auto target_id = _vocabulary.getUid(target);

  uint32_t step = std::strtoul(/* str= */ input.column(_step_col).data(),
                               /* str_end= */ nullptr, /* base= */ 10);

  auto label_id = _vocabulary.vocabSize() * step + target_id;

  vec.addSparseFeatureToSegment(/* index= */ label_id, /* value= */ 1.0);

  return nullptr;
}

std::string SequenceTargetBlock::className(uint32_t label_id) {
  uint32_t target_id = label_id % _vocabulary.vocabSize();
  return _vocabulary.getString(target_id);
}

std::string SequenceTargetBlock::classNameAtStep(const BoltVector& activations,
                                                 uint32_t step) {
  if (!activations.isDense()) {
    throw std::invalid_argument(
        "Cannot get sequence element name from dense output");
  }
  auto begin = step * _vocabulary.vocabSize();
  auto end = begin + _vocabulary.vocabSize();

  uint32_t arg_max = 0;
  float max_act = -std::numeric_limits<float>::max();
  for (uint32_t neuron = begin; neuron < end; neuron++) {
    if (activations.activations[neuron] > max_act) {
      arg_max = neuron;
      max_act = activations.activations[neuron];
    }
  }

  return className(arg_max);
}

std::vector<ColumnIdentifier*>
SequenceTargetBlock::concreteBlockColumnIdentifiers() {
  return {&_target_col, &_step_col};
}

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::SequenceBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::SequenceTargetBlock)