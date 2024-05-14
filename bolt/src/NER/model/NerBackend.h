#pragma once

#include <bolt/src/NER/model/utils.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Pipeline.h>
#include <dataset/src/DataSource.h>
#include <licensing/src/CheckLicense.h>
#include <string>
#include <unordered_map>

namespace thirdai::bolt {

class NerModelInterface {
 public:
  virtual ~NerModelInterface() = default;

  virtual std::vector<PerTokenListPredictions> getTags(
      std::vector<std::vector<std::string>> tokens, uint32_t top_k) const = 0;

  virtual metrics::History train(
      const dataset::DataSourcePtr& train_data, float learning_rate,
      uint32_t epochs, size_t batch_size,
      const std::vector<std::string>& train_metrics,
      const dataset::DataSourcePtr& val_data,
      const std::vector<std::string>& val_metrics) const = 0;

  virtual ar::ConstArchivePtr toArchive() const = 0;

  virtual std::string type() const = 0;

  virtual std::unordered_map<std::string, uint32_t> getTagToLabel() const = 0;
};
}  // namespace thirdai::bolt