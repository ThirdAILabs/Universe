// #include <eigen/unsupported/Eigen/CXX11/Tensor>

#include "wrappers/src/EigenDenseWrapper.h"

using RowMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

inline RowMatrixXf eigenMult(const Eigen::Ref<const RowMatrixXf>& m1,
                             const Eigen::Ref<const RowMatrixXf>& m2) {
  return m1 * m2;
}

// outputTensor = inputTensor
// .extract_image_patches(kern_w, kern_h, stride_w, stride_h, dilation_w,
// dilation_h, padding) .reshape(Eigen::array<int, 2>({patch_count,
// kern_w*kern_h})) .contract(kernalTensor.reshape(Eigen::array<int,
// 2>({kern_w*kern_h, kern_count})), {Eigen::IndexPair < int > (1, 0)})
// .reshape(Eigen::array<int, 3>({ output_w, output_h, kern_count }));