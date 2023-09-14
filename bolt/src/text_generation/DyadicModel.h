#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/text_generation/GenerativeModel.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/DyadicInterval.h>
#include <dataset/src/DataSource.h>

namespace thirdai::bolt {

class DyadicModel final : public GenerativeBackend {
 public:
  explicit DyadicModel(bolt::ModelPtr model);

  bolt::TensorPtr nextTokenProbs(
      std::vector<std::vector<uint32_t>> tokens) final;

  metrics::History train(const dataset::DataSourcePtr& train_data,
                         float learning_rate, uint32_t epochs,
                         size_t batch_size,
                         const std::vector<std::string>& train_metrics,
                         const dataset::DataSourcePtr& val_data,
                         const std::vector<std::string>& val_metrics,
                         const DistributedCommPtr& comm) final;

  bolt::ModelPtr getBoltModel() final { return _model; }

 private:
  data::Loader getDataLoader(const dataset::DataSourcePtr& data,
                             size_t batch_size, bool shuffle);

  bolt::ModelPtr _model;
  std::shared_ptr<data::DyadicInterval> _dyadic_transform;
  data::OutputColumnsList _bolt_inputs;
  size_t _vocab_size;

  DyadicModel() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<GenerativeBackend>(this), _model,
            _dyadic_transform, _bolt_inputs, _vocab_size);
  }
};

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::DyadicModel)