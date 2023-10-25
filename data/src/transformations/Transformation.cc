#include "Transformation.h"
#include <data/src/transformations/Binning.h>
#include <data/src/transformations/CategoricalTemporal.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/CountTokens.h>
#include <data/src/transformations/CrossColumnPairgrams.h>
#include <data/src/transformations/Date.h>
#include <data/src/transformations/DeduplicateTokens.h>
#include <data/src/transformations/DyadicInterval.h>
#include <data/src/transformations/EncodePosition.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/Graph.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/Recurrence.h>
#include <data/src/transformations/RegressionBinning.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/StringIDLookup.h>
#include <data/src/transformations/Tabular.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <proto/sequence.pb.h>
#include <proto/string_cast.pb.h>
#include <proto/transformations.pb.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::data {

TransformationPtr Transformation::fromProto(
    const proto::data::Transformation& transformation) {
  switch (transformation.type_case()) {
    case proto::data::Transformation::kBinning:
      return std::make_shared<BinningTransformation>(transformation.binning());

    case proto::data::Transformation::kCategoricalTemporal:
      return std::make_shared<CategoricalTemporal>(
          transformation.categorical_temporal());

    case proto::data::Transformation::kColdStart:
      return std::make_shared<ColdStartTextAugmentation>(
          transformation.cold_start());

    case proto::data::Transformation::kCountTokens:
      return std::make_shared<CountTokens>(transformation.count_tokens());

    case proto::data::Transformation::kCrossColumnPairgrams:
      return std::make_shared<CrossColumnPairgrams>(
          transformation.cross_column_pairgrams());

    case proto::data::Transformation::kDate:
      return std::make_shared<Date>(transformation.date());

    case proto::data::Transformation::kDeduplicateTokens:
      return std::make_shared<DeduplicateTokens>(
          transformation.deduplicate_tokens());

    case proto::data::Transformation::kDyadicInterval:
      return std::make_shared<DyadicInterval>(transformation.dyadic_interval());

    case proto::data::Transformation::kFeatureHash:
      return std::make_shared<FeatureHash>(transformation.feature_hash());

    case proto::data::Transformation::kGraphBuilder:
      return std::make_shared<GraphBuilder>(transformation.graph_builder());

    case proto::data::Transformation::kHashedPositionEncoding:
      return std::make_shared<HashPositionTransform>(
          transformation.hashed_position_encoding());

    case proto::data::Transformation::kMachLabel:
      return std::make_shared<MachLabel>(transformation.mach_label());

    case proto::data::Transformation::kNeighborFeatures:
      return std::make_shared<NeighborFeatures>(
          transformation.neighbor_features());

    case proto::data::Transformation::kNeighborIds:
      return std::make_shared<NeighborIds>(transformation.neighbor_ids());

    case proto::data::Transformation::kOffsetPositionEncoding:
      return std::make_shared<OffsetPositionTransform>(
          transformation.offset_position_encoding());

    case proto::data::Transformation::kRecurrenceAugmentation:
      return std::make_shared<Recurrence>(
          transformation.recurrence_augmentation());

    case proto::data::Transformation::kRegressionBinning:
      return std::make_shared<RegressionBinning>(
          transformation.regression_binning());

    case proto::data::Transformation::kStringCast:
      return stringCastFromProto(transformation.string_cast());

    case proto::data::Transformation::kStringConcat:
      return std::make_shared<StringConcat>(transformation.string_concat());

    case proto::data::Transformation::kStringHash:
      return std::make_shared<StringHash>(transformation.string_hash());

    case proto::data::Transformation::kStringIdLookup:
      return std::make_shared<StringIDLookup>(
          transformation.string_id_lookup());

    case proto::data::Transformation::kTabular:
      return std::make_shared<Tabular>(transformation.tabular());

    case proto::data::Transformation::kTextTokenizer:
      return std::make_shared<TextTokenizer>(transformation.text_tokenizer());

    case proto::data::Transformation::kPipeline:
      return std::make_shared<Pipeline>(transformation.pipeline());

    default:
      throw std::runtime_error("Invalid transformation type in fromProto.");
  }
}

std::string Transformation::serialize() const {
  auto* proto = toProto();
  auto binary = proto->SerializeAsString();
  delete proto;

  return binary;
}

std::shared_ptr<Transformation> Transformation::deserialize(
    const std::string& binary) {
  proto::data::Transformation transformation;
  transformation.ParseFromString(binary);

  return Transformation::fromProto(transformation);
}

}  // namespace thirdai::data