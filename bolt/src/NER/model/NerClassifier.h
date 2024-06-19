#pragma once

#include <bolt/src/NER/model/utils.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Pipeline.h>
#include <dataset/src/DataSource.h>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace thirdai::bolt::NER {

using PerTokenPredictions = std::vector<std::pair<std::string, float>>;
using PerTokenListPredictions = std::vector<PerTokenPredictions>;

void applyPunctAndStopWordFilter(const std::string& token,
                                 PerTokenPredictions& predicted_tags,
                                 const std::string& default_tag);

class NerClassifier {
 public:
  NerClassifier(bolt::ModelPtr model, data::OutputColumnsList bolt_inputs,
                data::TransformationPtr train_transforms,
                data::TransformationPtr inference_transforms,
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
      std::vector<std::vector<std::string>> tokens, uint32_t top_k,
      const std::unordered_map<uint32_t, std::string>& label_to_tag_map,
      const std::unordered_map<std::string, uint32_t>& tag_to_label_map) const;

 private:
  bolt::ModelPtr _bolt_model;
  data::TransformationPtr _train_transforms, _inference_transforms;
  data::OutputColumnsList _bolt_inputs;
  std::string _tokens_column;
  std::string _tags_column;
};

using NerClassifierPtr = std::shared_ptr<NerClassifier>;

}  // namespace thirdai::bolt::NER