#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/seismic/SeismicBase.h>
#include <bolt/src/seismic/SeismicLabels.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/DistributedComm.h>
#include <bolt/src/train/trainer/Trainer.h>

namespace thirdai::bolt::seismic {

class SeismicEmbedding final : public SeismicBase {
 public:
  static std::shared_ptr<SeismicEmbedding> make(size_t subcube_shape,
                                                size_t patch_shape,
                                                size_t embedding_dim,
                                                const std::string& model_size,
                                                std::optional<size_t> max_pool);

  SeismicEmbedding(InputShapeData input_shape_data, ModelPtr model);

  metrics::History trainOnPatches(
      const NumpyArray& subcubes,
      const std::vector<SubcubeMetadata>& subcube_metadata, float learning_rate,
      size_t batch_size, const std::vector<callbacks::CallbackPtr>& callbacks,
      std::optional<uint32_t> log_interval, const DistributedCommPtr& comm);

 private:
  Dataset makeLabelBatches(const std::vector<SubcubeMetadata>& subcube_metadata,
                           size_t batch_size) const;

  static ModelPtr buildModel(size_t n_patches, size_t patch_dim,
                             size_t embedding_dim, size_t n_output_classes,
                             float output_sparsity);

  // The subcube and patch shapes are stored as tuples because we want to
  // support a case where the subcubes are 2D, with a shape like (1, 10, 10),
  // but we sill want these 2D slices that are nearby in the x-axis to share
  // labels. Thus the labels are always associated with 3D cubes in space, which
  // ensures that overlapping subcubes in any of the three axes will share
  // labels.
  // TODO(Nicholas): support for list of label cube dims for different
  // granularities.
  size_t _label_cube_dim = 32;

  size_t _n_output_classes;

  SeismicEmbedding() : SeismicBase({{}, {}, {}}, {}) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::bolt::seismic