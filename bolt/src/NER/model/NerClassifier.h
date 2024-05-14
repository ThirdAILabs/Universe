
#include <bolt/src/NER/model/utils.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Pipeline.h>
#include <dataset/src/DataSource.h>
#include <memory>

namespace thirdai::bolt {
class NerClassifier {
 public:
  NerClassifier(bolt::ModelPtr model, data::OutputColumnsList bolt_inputs,
                data::PipelinePtr train_transforms,
                data::PipelinePtr inference_transforms,
                std::string tokens_column)
      : _bolt_model(std::move(model)),
        _train_transforms(std::move(train_transforms)),
        _inference_transforms(std::move(inference_transforms)),
        _bolt_inputs(std::move(bolt_inputs)),
        _tokens_column(std::move(tokens_column)) {}

  metrics::History train(const dataset::DataSourcePtr& train_data,
                         float learning_rate, uint32_t epochs,
                         size_t batch_size,
                         const std::vector<std::string>& train_metrics,
                         const dataset::DataSourcePtr& val_data,
                         const std::vector<std::string>& val_metrics) const {
    auto train_dataset =
        getDataLoader(train_data, batch_size, /* shuffle= */ true).all();
    bolt::LabeledDataset val_dataset;
    if (val_data) {
      val_dataset =
          getDataLoader(val_data, batch_size, /* shuffle= */ false).all();
    }
    auto train_data_input = train_dataset.first;
    auto train_data_label = train_dataset.second;

    Trainer trainer(_bolt_model);
    // We cannot use train_with_dataset_loader, since it is using the older
    // dataset::DatasetLoader while dyadic model is using data::Loader
    for (uint32_t e = 0; e < epochs; e++) {
      trainer.train_with_metric_names(
          train_dataset, learning_rate, 1, train_metrics, val_dataset,
          val_metrics, /* steps_per_validation= */ std::nullopt,
          /* use_sparsity_in_validation= */ false, /* callbacks= */ {},
          /* autotune_rehash_rebuild= */ false, /* verbose= */ true);
    }
    return trainer.getHistory();
  }

  data::Loader getDataLoader(const dataset::DataSourcePtr& data,
                             size_t batch_size, bool shuffle) const {
    auto data_iter =
        data::JsonIterator::make(data, {_tokens_column, _tokens_column}, 1000);

    return data::Loader(data_iter, _train_transforms, nullptr, _bolt_inputs,
                        {data::OutputColumns(_tokens_column)},
                        /* batch_size= */ batch_size,
                        /* shuffle= */ shuffle, /* verbose= */ true,
                        /* shuffle_buffer_size= */ 20000);
  }

  std::vector<PerTokenListPredictions> getTags(
      std::vector<std::vector<std::string>> tokens, uint32_t top_k) const {
    return thirdai::bolt::getTags(std::move(tokens), top_k, _tokens_column,
                                  _inference_transforms, _bolt_inputs,
                                  _bolt_model);
  }

 private:
  bolt::ModelPtr _bolt_model;
  data::PipelinePtr _train_transforms, _inference_transforms;
  data::OutputColumnsList _bolt_inputs;
  std::string _tokens_column;
};

using NerClassifierPtr = std::shared_ptr<NerClassifier>;
}  // namespace thirdai::bolt