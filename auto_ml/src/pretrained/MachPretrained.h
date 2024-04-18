#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/metrics/Metric.h>
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

namespace thirdai::automl {

class MachPretrained {
 public:
  MachPretrained(std::string input_column, dataset::TextTokenizerPtr tokenizer,
                 size_t vocab_size, size_t emb_dim, size_t output_dim,
                 size_t n_models);

  std::vector<bolt::metrics::History> train(
      const data::ColumnMapIteratorPtr& train_data, size_t epochs,
      size_t batch_size, float learning_rate,
      const data::ColumnMapIteratorPtr& val_data);

  std::vector<std::vector<uint32_t>> decodeHashes(
      std::vector<std::string> phrases, size_t hashes_per_model);

 private:
  data::TransformationPtr buildPipeline() {
    return data::Pipeline::make()
        ->then(_tokenizer)
        ->then(_nwp)
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
  std::shared_ptr<data::NextWordPrediction> _nwp;

  std::string _input_column;

  std::string _source_column = "__source__";
  std::string _target_column = "__target__";
};

}  // namespace thirdai::automl