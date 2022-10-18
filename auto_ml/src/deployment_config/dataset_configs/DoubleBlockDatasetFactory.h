
#pragma once

#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <dataset/src/blocks/BlockInterface.h>

namespace thirdai::automl::deployment {

class DoubleBlockDatasetFactory final : public DatasetLoaderFactory {
 public:
  void preprocessDataset(
      const std::shared_ptr<dataset::DataLoader>& data_loader,
      std::optional<uint64_t> max_in_memory_batches) final {
    (void)data_loader;
    (void)max_in_memory_batches;
  }

  DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataLoader> data_loader, bool training) final {
    (void)data_loader;
    (void)training;
    return nullptr;
  }

  std::vector<BoltVector> featurizeInput(const std::string& input) final {
    (void)input;
    return {};
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const std::vector<std::string>& inputs) final {
    (void)inputs;
    return {};
  }

    std::vector<bolt::InputPtr> getInputNodes() final {
        return {};
    }

private:
    friend class cereal::access;

    template <class Archive>
    void serialize(Archive &archive) {
        archive();
    }


};

}  // namespace thirdai::automl::deployment