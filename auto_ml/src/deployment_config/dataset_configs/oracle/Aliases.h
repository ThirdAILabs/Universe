/**
 * 2022-10-12
 *
 * Since we are currently transitioning between the standalone sequential
 * classifier and the model pipeline counterpart, we want to reuse classes
 * without reorganizing the repository too much.
 *
 * This file allows us to have consistent and concise aliases for these types
 * to be used within the thirdai::automl::deployment namespace.
 */

#pragma once

#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/auto_classifiers/sequential_classifier/ConstructorUtilityTypes.h>
#include <bolt/src/auto_classifiers/sequential_classifier/SequentialUtils.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <map>
#include <unordered_map>

namespace thirdai::automl::deployment {

using CategoricalType = bolt::sequential_classifier::CategoricalDataType;
using MetadataConfig = bolt::sequential_classifier::CategoricalMetadataConfig;

using DataType = bolt::sequential_classifier::DataType;
using ColumnDataTypes = std::map<std::string, DataType>;

using TemporalConfig = bolt::sequential_classifier::TemporalConfig;
using UserProvidedTemporalRelationships =
    std::map<std::string,
             std::vector<std::variant<std::string, TemporalConfig>>>;
using TemporalRelationships =
    std::map<std::string, std::vector<TemporalConfig>>;

using ColumnNumberMap = bolt::sequential_classifier::ColumnNumberMap;
using ColumnNumberMapPtr = std::shared_ptr<ColumnNumberMap>;

using ColumnVocabularies =
    std::unordered_map<std::string, dataset::ThreadSafeVocabularyPtr>;
}  // namespace thirdai::automl::deployment