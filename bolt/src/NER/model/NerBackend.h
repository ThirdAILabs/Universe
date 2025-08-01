#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <licensing/src/CheckLicense.h>
#include <unordered_map>
#include <utility>

namespace thirdai::bolt::NER {
class NerModelInterface {
 public:
  virtual ~NerModelInterface() = default;

  virtual std::vector<std::vector<std::vector<std::pair<std::string, float>>>>
  getTags(std::vector<std::vector<std::string>> tokens,
          uint32_t top_k) const = 0;

  virtual metrics::History train(
      const dataset::DataSourcePtr& train_data, float learning_rate,
      uint32_t epochs, size_t batch_size,
      const std::vector<std::string>& train_metrics,
      const dataset::DataSourcePtr& val_data,
      const std::vector<std::string>& val_metrics) = 0;

  virtual ar::ConstArchivePtr toArchive() const = 0;

  virtual std::string type() const = 0;

  virtual std::unordered_map<std::string, uint32_t> getTagToLabel() = 0;

  virtual std::string getTokensColumn() const = 0;

  virtual std::string getTagsColumn() const = 0;

  virtual bolt::ModelPtr getBoltModel() = 0;
};

}  // namespace thirdai::bolt::NER