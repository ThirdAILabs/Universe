#include "RecurrenceAugmentation.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/BlockInterface.h>

namespace thirdai::dataset {

RecurrenceAugmentation::RecurrenceAugmentation(ColumnIdentifier sequence_column,
                                               char delimiter,
                                               uint32_t max_recurrence,
                                               uint32_t vocab_size,
                                               uint32_t input_vector_index,
                                               uint32_t label_vector_index)
    : _sequence_column(std::move(sequence_column)),
      _delimiter(delimiter),
      _max_recurrence(max_recurrence),
      _input_vector_index(input_vector_index),
      _label_vector_index(label_vector_index),
      _vocab(vocab_size + 1) {
  /*
    We will fix the vocabulary for better parallelism when the vocabulary is
    full. The EOS token is registered at construction time to avoid the case
    where we cannot fix the vocabulary because EOS is not found in the dataset,
    e.g. if all target sequences are max_recurrence elements long.
  */
  _vocab.getUid(EOS);
}

void RecurrenceAugmentation::prepareForBatch(
    ColumnarInputBatch& incoming_batch) {
  (void)incoming_batch;
  // Do nothing.
}

std::vector<std::vector<BoltVector>> RecurrenceAugmentation::augment(
    std::vector<SegmentedFeatureVectorPtr>&& builders,
    ColumnarInputSample& input_sample) {
  auto target_sequence = sequence(input_sample);
  auto element_ids = elementIds(target_sequence);

  std::vector<std::vector<BoltVector>> vectors(builders.size());

  for (uint32_t vector_id = 0; vector_id < builders.size(); vector_id++) {
    if (vector_id == _input_vector_index) {
      vectors.at(vector_id) = augmentInputVectors(
          /* builder= */ *builders.at(vector_id),
          /* elements= */ element_ids);
    } else if (vector_id == _label_vector_index) {
      vectors.at(vector_id) = augmentLabelVectors(
          /* builder= */ *builders.at(vector_id),
          /* elements= */ element_ids);
    } else {
      vectors.at(vector_id) = replicateOtherVectors(
          /* builder= */ *builders.at(vector_id),
          /* size= */ element_ids.size());
    }
  }

  return vectors;
}

std::vector<BoltVector> RecurrenceAugmentation::augmentInputVectors(
    SegmentedFeatureVector& builder, std::vector<uint32_t> elements) {
  std::vector<BoltVector> vectors(elements.size());
  vectors[0] = builder.toBoltVector();
  for (uint32_t i = 0; i < elements.size() - 1; i++) {
    builder.addSparseFeatureToSegment(/* index= */ elements[i],
                                      /* value= */ 1.0);
    vectors[i + 1] = builder.toBoltVector();
  }
  return vectors;
}

std::vector<BoltVector> RecurrenceAugmentation::augmentLabelVectors(
    SegmentedFeatureVector& builder, std::vector<uint32_t> elements) {
  auto vector = builder.toBoltVector();
  if (vector.len > 0) {
    throw std::invalid_argument(
        "RecurrenceAugmentation expects to be the exclusive feature in the "
        "label vector.");
  }

  std::vector<BoltVector> vectors(elements.size());
  for (uint32_t i = 0; i < elements.size(); i++) {
    vectors[i] = BoltVector::singleElementSparseVector(elements[i]);
  }

  return vectors;
}

std::vector<BoltVector> RecurrenceAugmentation::replicateOtherVectors(
    SegmentedFeatureVector& builder, uint32_t size) {
  std::vector<BoltVector> vectors(size);
  for (auto& bolt_vec : vectors) {
    bolt_vec = builder.toBoltVector();
  }
  return vectors;
}

uint32_t RecurrenceAugmentation::elementIdAtStep(const BoltVector& output,
                                                 uint32_t step) {
  if (!output.isDense()) {
    throw std::invalid_argument(
        "Cannot get sequence element name from dense output");
  }
  auto begin = step * _vocab.maxSize().value();
  auto end = begin + _vocab.maxSize().value();

  uint32_t arg_max = 0;
  float max_act = -std::numeric_limits<float>::max();
  for (uint32_t neuron = begin; neuron < end; neuron++) {
    if (output.activations[neuron] > max_act) {
      arg_max = neuron;
      max_act = output.activations[neuron];
    }
  }

  return arg_max;
}

std::string RecurrenceAugmentation::elementString(uint32_t element_id) {
  uint32_t element_id_without_position = element_id % _vocab.maxSize().value();
  return _vocab.getString(element_id_without_position);
}

bool RecurrenceAugmentation::isEOS(uint32_t element_id) {
  return elementString(element_id) == EOS;
}

std::vector<std::string_view> RecurrenceAugmentation::sequence(
    ColumnarInputSample& input_sample) const {
  auto sequence =
      text::split(input_sample.column(_sequence_column), _delimiter);
  if (sequence.size() < _max_recurrence) {
    sequence.push_back(std::string_view(EOS, EOS_SIZE));
  }

  if (sequence.size() > _max_recurrence) {
    sequence.resize(_max_recurrence);
  }
  return sequence;
}

std::vector<uint32_t> RecurrenceAugmentation::elementIds(
    const std::vector<std::string_view>& sequence) {
  std::vector<uint32_t> element_ids(sequence.size());
  for (uint32_t i = 0; i < element_ids.size(); i++) {
    uint32_t offset = i * _vocab.maxSize().value();
    element_ids[i] = offset + _vocab.getUid(std::string(sequence[i]));
  }
  return element_ids;
}

template void RecurrenceAugmentation::serialize(cereal::BinaryInputArchive&);
template void RecurrenceAugmentation::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void RecurrenceAugmentation::serialize(Archive& archive) {
  archive(cereal::base_class<Augmentation>(this), _sequence_column, _delimiter,
          _max_recurrence, _input_vector_index, _label_vector_index, _vocab);
}

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::RecurrenceAugmentation)