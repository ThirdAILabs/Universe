#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/metrics/Metric.h>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/NextWordPrediction.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/mach/MachIndex.h>
#include <memory>

namespace thirdai::automl {

class SpladeMach : public std::enable_shared_from_this<SpladeMach> {
 public:
  SpladeMach(std::string input_column, std::vector<bolt::ModelPtr> models,
             std::vector<data::MachIndexPtr> indexes,
             dataset::TextTokenizerPtr tokenizer);

  std::vector<bolt::metrics::History> train(
      const data::ColumnMapIteratorPtr& train_data, size_t epochs,
      size_t batch_size, float learning_rate,
      const data::ColumnMapIteratorPtr& val_data);

  std::vector<std::vector<uint32_t>> getTopHashBuckets(
      std::vector<std::string> phrases, size_t hashes_per_model);

  std::vector<uint32_t> getTopTokens(std::string phrase, size_t num_tokens,
                                     size_t num_buckets_to_decode);

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<SpladeMach> fromArchive(const ar::Archive& archive);

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<SpladeMach> load(const std::string& filename);

  static std::shared_ptr<SpladeMach> load_stream(std::istream& input_stream);

 private:
  data::TransformationPtr buildPipeline() {
    return data::Pipeline::make()
        ->then(_tokenizer)
        ->then(std::make_shared<data::NextWordPrediction>(
            _source_column, _source_column, _target_column))
        ->then(
            std::make_shared<data::MachLabel>(_target_column, _target_column));
  }

  std::shared_ptr<data::Loader> getDataLoader(
      data::ColumnMapIteratorPtr data_iter, data::StatePtr state,
      size_t batch_size) {
    return std::make_shared<data::Loader>(
        std::move(data_iter), buildPipeline(), std::move(state),
        data::OutputColumnsList{data::OutputColumns(_source_column)},
        data::OutputColumnsList{data::OutputColumns(_target_column)},
        batch_size, /*shuffle=*/true, /*verbose=*/true,
        /*shuffle_buffer_size=*/1000000);
  }

  std::vector<bolt::ModelPtr> _models;
  std::vector<data::MachIndexPtr> _indexes;

  data::TextTokenizerPtr _tokenizer;

  std::string _input_column;
  //   size_t _vocab_size;

  const std::string _source_column = "__source__";
  const std::string _target_column = "__target__";
};

using SpladeMachPtr = std::shared_ptr<SpladeMach>;

}  // namespace thirdai::automl