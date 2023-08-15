#include "Transformation.h"
#include <data/src/transformations/Binning.h>
#include <data/src/transformations/CategoricalTemporal.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/Date.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/StringIDLookup.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <proto/string_cast.pb.h>
#include <proto/transformations.pb.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::data {

char getDelimiter(const proto::data::StringCast& cast) {
  if (!cast.has_delimiter()) {
    throw std::runtime_error("Expected delimiter for cast to array column.");
  }
  return cast.delimiter();
}

std::string getFormat(const proto::data::StringCast& cast) {
  if (!cast.has_format()) {
    throw std::runtime_error(
        "Expected time format for cast to timestamp column.");
  }
  return cast.format();
}

TransformationPtr stringCastFromProto(const proto::data::StringCast& cast) {
  std::optional<size_t> dim;
  if (cast.has_dim()) {
    dim = cast.dim();
  }

  switch (cast.target()) {
    case proto::data::StringCast::TOKEN:
      return std::make_shared<StringToToken>(cast.input_column(),
                                             cast.output_column(), dim);

    case proto::data::StringCast::TOKEN_ARRAY:
      return std::make_shared<StringToTokenArray>(
          cast.input_column(), cast.output_column(), getDelimiter(cast), dim);

    case proto::data::StringCast::DECIMAL:
      return std::make_shared<StringToDecimal>(cast.input_column(),
                                               cast.output_column());

    case proto::data::StringCast::DECIMAL_ARRAY:
      return std::make_shared<StringToDecimalArray>(
          cast.input_column(), cast.output_column(), getDelimiter(cast), dim);

    case proto::data::StringCast::TIMESTAMP:
      return std::make_shared<StringToTimestamp>(
          cast.input_column(), cast.output_column(), getFormat(cast));

    default:
      throw std::runtime_error("Invalid string cast target type in fromProto.");
  }
}

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

    case proto::data::Transformation::kDate:
      return std::make_shared<Date>(transformation.date());

    case proto::data::Transformation::kFeatureHash:
      return std::make_shared<FeatureHash>(transformation.feature_hash());

    case proto::data::Transformation::kMachLabel:
      return std::make_shared<MachLabel>(transformation.mach_label());

    case proto::data::Transformation::kStringCast:
      return stringCastFromProto(transformation.string_cast());

    case proto::data::Transformation::kStringConcat:
      return std::make_shared<StringConcat>(transformation.string_concat());

    case proto::data::Transformation::kStringHash:
      return std::make_shared<StringHash>(transformation.string_hash());

    case proto::data::Transformation::kStringIdLookup:
      return std::make_shared<StringIDLookup>(
          transformation.string_id_lookup());

    case proto::data::Transformation::kTextTokenizer:
      return std::make_shared<TextTokenizer>(transformation.text_tokenizer());

    case proto::data::Transformation::kList:
      return std::make_shared<TransformationList>(transformation.list());

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