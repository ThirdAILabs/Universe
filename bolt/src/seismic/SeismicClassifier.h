#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/seismic/SeismicBase.h>

namespace thirdai::bolt::seismic {

class SeismicClassifier final : public SeismicBase {
 public:
  SeismicClassifier(const std::shared_ptr<SeismicBase>& emb_model,
                    size_t n_classes);

  metrics::History trainOnPatches(
      const NumpyArray& subcubes, std::vector<std::vector<uint32_t>> labels,
      float learning_rate, size_t batch_size,
      const std::vector<callbacks::CallbackPtr>& callbacks,
      std::optional<uint32_t> log_interval, const DistributedCommPtr& comm);

  void save(const std::string& filename) const final;

  void save_stream(std::ostream& output) const;

  static std::shared_ptr<SeismicClassifier> load(const std::string& filename);

  static std::shared_ptr<SeismicClassifier> load_stream(std::istream& input);

 private:
  Dataset makeLabelbatches(std::vector<std::vector<uint32_t>> labels,
                           size_t batch_size);

  static ModelPtr addClassifierHead(const ModelPtr& emb_model,
                                    size_t n_classes);

  SeismicClassifier() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::bolt::seismic