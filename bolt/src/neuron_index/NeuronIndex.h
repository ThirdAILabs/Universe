#pragma once

#include <cereal/access.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <memory>
#include <unordered_set>

namespace thirdai::bolt {

class NeuronIndex {
 public:
  virtual void query(const BoltVector& input, BoltVector& output,
                     const BoltVector* labels) const = 0;

  virtual void buildIndex(const std::vector<float>& weights, uint32_t dim,
                          bool use_new_seed) = 0;

  virtual void autotuneForNewSparsity(uint32_t dim, uint32_t prev_dim,
                                      float sparsity,
                                      bool experimental_autotune) = 0;

  virtual void insertLabelsIfNotFound() {}

  virtual void summarize(std::ostream& summary) const = 0;

  virtual ar::ConstArchivePtr toArchive() const {
    auto map = ar::Map::make();
    map->set("type", ar::str("none"));
    return map;
  }

  static std::shared_ptr<NeuronIndex> fromArchive(const ar::Archive& archive);

  virtual ~NeuronIndex() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using NeuronIndexPtr = std::shared_ptr<NeuronIndex>;

}  // namespace thirdai::bolt