#include "Transformation.h"
#include <archive/src/Archive.h>
#include <data/src/transformations/AddMachMemorySamples.h>
#include <data/src/transformations/Binning.h>
#include <data/src/transformations/CategoricalTemporal.h>
#include <data/src/transformations/CountTokens.h>
#include <data/src/transformations/CrossColumnPairgrams.h>
#include <data/src/transformations/Date.h>
#include <data/src/transformations/DeduplicateTokens.h>
#include <data/src/transformations/DyadicInterval.h>
#include <data/src/transformations/EncodePosition.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/Graph.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/NextWordPrediction.h>
#include <data/src/transformations/NumericalTemporal.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/Recurrence.h>
#include <data/src/transformations/RegressionBinning.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/StringIDLookup.h>
#include <data/src/transformations/Tabular.h>
#include <data/src/transformations/TextCompat.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/ner/NerTokenizationUnigram.h>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace thirdai::data {

// NOLINTNEXTLINE clang-tidy doesn't like macros.
#define HANDLE_TYPE(transformation)                   \
  if (type == transformation::type()) {               \
    return std::make_shared<transformation>(archive); \
  }

TransformationPtr Transformation::fromArchive(const ar::Archive& archive) {
  std::string type = archive.str("type");

  HANDLE_TYPE(BinningTransformation)
  HANDLE_TYPE(CategoricalTemporal)
  HANDLE_TYPE(NumericalTemporal)
  HANDLE_TYPE(CountTokens)
  HANDLE_TYPE(CrossColumnPairgrams)
  HANDLE_TYPE(Date)
  HANDLE_TYPE(DeduplicateTokens)
  HANDLE_TYPE(DyadicInterval)
  HANDLE_TYPE(FeatureHash)
  HANDLE_TYPE(GraphBuilder)
  HANDLE_TYPE(NeighborIds)
  HANDLE_TYPE(NeighborFeatures)
  HANDLE_TYPE(HashPositionTransform)
  HANDLE_TYPE(OffsetPositionTransform)
  HANDLE_TYPE(MachLabel)
  HANDLE_TYPE(Pipeline)
  HANDLE_TYPE(Recurrence)
  HANDLE_TYPE(RegressionBinning)
  HANDLE_TYPE(CastToValue<uint32_t>)
  HANDLE_TYPE(CastToValue<float>)
  HANDLE_TYPE(CastToValue<int64_t>)
  HANDLE_TYPE(CastToArray<uint32_t>)
  HANDLE_TYPE(CastToArray<float>)
  HANDLE_TYPE(StringConcat)
  HANDLE_TYPE(StringHash)
  HANDLE_TYPE(StringIDLookup)
  HANDLE_TYPE(Tabular)
  HANDLE_TYPE(TextTokenizer)
  HANDLE_TYPE(AddMachMemorySamples)
  HANDLE_TYPE(NextWordPrediction)
  HANDLE_TYPE(NerTokenizerUnigram)
  HANDLE_TYPE(TextCompat)

  throw std::runtime_error("Invalid transformation type '" + type +
                           "' in fromArchive.");
}

std::string Transformation::serialize() const {
  std::stringstream buffer;
  ar::serialize(toArchive(), buffer);
  return buffer.str();
}

std::shared_ptr<Transformation> Transformation::deserialize(
    const std::string& bytes) {
  std::istringstream input(bytes);
  auto archive = ar::deserialize(input);
  return fromArchive(*archive);
}

}  // namespace thirdai::data