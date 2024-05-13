
#pragma once

#include <bolt/src/NER/model/NerBackend.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/text_generation/GenerativeModel.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/DyadicInterval.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/DataSource.h>
#include <unordered_map>

namespace thirdai::bolt {

class NerBoltModel final : public NerBackend {
 public:
  std::string type() const final { return "pretrained_ner"; }
  NerBoltModel(bolt::ModelPtr model,
               std::unordered_map<std::string, uint32_t> tag_to_label);

  NerBoltModel(std::string& pretrained_model_path, std::string token_column,
               std::string tag_column,
               std::unordered_map<std::string, uint32_t> tag_to_label);

  std::vector<PerTokenListPredictions> getTags(
      std::vector<std::vector<std::string>> tokens, uint32_t top_k) final;

  metrics::History train(const dataset::DataSourcePtr& train_data,
                         float learning_rate, uint32_t epochs,
                         size_t batch_size,
                         const std::vector<std::string>& train_metrics,
                         const dataset::DataSourcePtr& val_data,
                         const std::vector<std::string>& val_metrics) final;

  ar::ConstArchivePtr toArchive() const final;

  std::unordered_map<std::string, uint32_t> getTagToLabel() final {
    return _tag_to_label;
  }

  static std::shared_ptr<NerBoltModel> fromArchive(const ar::Archive& archive);

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<NerBoltModel> load(const std::string& filename);

  static std::shared_ptr<NerBoltModel> load_stream(std::istream& input_stream);

  NerBoltModel() = default;
  ~NerBoltModel() override = default;

 private:
  data::Loader getDataLoader(const dataset::DataSourcePtr& data,
                             size_t batch_size, bool shuffle);

  data::PipelinePtr getTransformations(bool inference);

  bolt::ModelPtr _bolt_model;
  data::PipelinePtr _train_transforms;
  data::PipelinePtr _inference_transforms;
  data::OutputColumnsList _bolt_inputs;
  std::unordered_map<std::string, uint32_t> _tag_to_label;

  std::string _source_column = "source";
  std::string _target_column = "target";

  const size_t _vocab_size = 50257;
};

}  // namespace thirdai::bolt