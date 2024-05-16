#pragma once

#include <bolt/src/NER/model/utils.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Pipeline.h>
#include <dataset/src/DataSource.h>
#include <memory>

namespace thirdai::bolt::NER {
class NerClassifier {
 public:
  NerClassifier(bolt::ModelPtr model, data::OutputColumnsList bolt_inputs,
                data::PipelinePtr train_transforms,
                data::PipelinePtr inference_transforms,
                std::string tokens_column, std::string tags_column)
      : _bolt_model(std::move(model)),
        _train_transforms(std::move(train_transforms)),
        _inference_transforms(std::move(inference_transforms)),
        _bolt_inputs(std::move(bolt_inputs)),
        _tokens_column(std::move(tokens_column)),
        _tags_column(std::move(tags_column)) {}

  metrics::History train(const dataset::DataSourcePtr& train_data,
                         float learning_rate, uint32_t epochs,
                         size_t batch_size,
                         const std::vector<std::string>& train_metrics,
                         const dataset::DataSourcePtr& val_data,
                         const std::vector<std::string>& val_metrics) const;

  data::Loader getDataLoader(const dataset::DataSourcePtr& data,
                             size_t batch_size, bool shuffle) const;

  std::vector<PerTokenListPredictions> getTags(
      std::vector<std::vector<std::string>> tokens, uint32_t top_k) const;

 private:
  bolt::ModelPtr _bolt_model;
  data::PipelinePtr _train_transforms, _inference_transforms;
  data::OutputColumnsList _bolt_inputs;
  std::string _tokens_column;
  std::string _tags_column;
};

using NerClassifierPtr = std::shared_ptr<NerClassifier>;
}  // namespace thirdai::bolt::NER