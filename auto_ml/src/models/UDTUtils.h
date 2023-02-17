#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/models/OutputProcessor.h>
#include <auto_ml/src/models/TrainEvalParameters.h>
#include <stdexcept>
namespace thirdai::automl::models {

static constexpr uint32_t DEFAULT_TRAIN_EVAL_BATCH_SIZE = 2048;
static constexpr const uint32_t TEXT_PAIRGRAM_WORD_LIMIT = 15;

TrainEvalParameters defaultTrainEvalParams(bool freeze_hash_tables);

void verifyDataTypesContainTarget(const data::ColumnDataTypes& data_types,
                                  const std::string& target);

/**
 * Returns the output processor to use to create the ModelPipeline. Also
 * returns a RegressionBinningStrategy if the output is a regression task as
 * this binning logic must be shared with the dataset pipeline.
 */
std::pair<OutputProcessorPtr, std::optional<dataset::RegressionBinningStrategy>>
getOutputProcessor(const data::ColumnDataTypes& data_types,
                   const std::string& target,
                   const std::optional<uint32_t>& n_target_classes);

}  // namespace thirdai::automl::models